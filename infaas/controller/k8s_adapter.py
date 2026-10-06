"""Kubernetes in place of EC2 (PLAN §2.2).

Worker pods carry `app=infaas-worker, gpu=<type>` and an `infaas-mode` label:
  static   created by the `infaas-worker-<type>` Deployments (like Lumina's workers)
  dynamic  created here, from the same Deployment's pod template, when the
           VM-Autoscaler adds a worker (the original's start_vm.sh). The
           Deployment's selector includes `infaas-mode: static`, so it never
           adopts these; an ownerReference to it lets `kubectl delete` of the
           Deployment clean them up.
Keeping the pod spec in the Deployment YAML means both modes run the same pod.
"""
from __future__ import annotations

import copy
import logging
import secrets
from dataclasses import dataclass
from typing import Dict, Optional

from infaas.common import config

log = logging.getLogger("k8s")

MODE_LABEL = "infaas-mode"


@dataclass
class PodInfo:
    name: str
    hw: str
    ip: str
    ready: bool
    phase: str
    managed: bool          # created by the autoscaler
    deleting: bool


class K8s:
    def __init__(self, namespace: str = config.NAMESPACE) -> None:
        from kubernetes import client, config as kcfg
        try:
            kcfg.load_incluster_config()
        except Exception:  # noqa: BLE001
            kcfg.load_kube_config()
        self.ns = namespace
        self.core = client.CoreV1Api()
        self.apps = client.AppsV1Api()
        self._client = client

    def list_workers(self) -> Dict[str, PodInfo]:
        pods = self.core.list_namespaced_pod(self.ns, label_selector=config.WORKER_LABEL)
        out = {}
        for p in pods.items:
            labels = p.metadata.labels or {}
            ready = any(cs.ready for cs in (p.status.container_statuses or []))
            out[p.metadata.name] = PodInfo(
                name=p.metadata.name, hw=labels.get("gpu", ""), ip=p.status.pod_ip or "",
                ready=ready and p.status.phase == "Running", phase=p.status.phase or "",
                managed=labels.get(MODE_LABEL) == "dynamic",
                deleting=p.metadata.deletion_timestamp is not None)
        return out

    def create_worker(self, hw: str) -> Optional[str]:
        """Start one more worker of this GPU type from its Deployment's template."""
        name = f"infaas-worker-{hw}"
        try:
            dep = self.apps.read_namespaced_deployment(name, self.ns)
        except Exception as e:  # noqa: BLE001
            log.error("cannot read deployment %s: %s", name, e)
            return None
        tmpl = dep.spec.template
        labels = dict(tmpl.metadata.labels or {})
        labels[MODE_LABEL] = "dynamic"
        pod_name = f"infaas-worker-{hw}-d{secrets.token_hex(3)}"
        c = self._client
        pod = c.V1Pod(
            api_version="v1", kind="Pod",
            metadata=c.V1ObjectMeta(
                name=pod_name, labels=labels,
                annotations=dict(tmpl.metadata.annotations or {}),
                owner_references=[c.V1OwnerReference(
                    api_version="apps/v1", kind="Deployment", name=dep.metadata.name,
                    uid=dep.metadata.uid, controller=False, block_owner_deletion=False)]),
            spec=copy.deepcopy(tmpl.spec))
        self.core.create_namespaced_pod(self.ns, pod)
        log.info("created worker pod %s (%s)", pod_name, hw)
        return pod_name

    def delete_pod(self, name: str) -> None:
        try:
            self.core.delete_namespaced_pod(name, self.ns, grace_period_seconds=10)
            log.info("deleted worker pod %s", name)
        except Exception as e:  # noqa: BLE001
            log.warning("delete %s: %s", name, e)
