"""All scalable profile services must have sidecar scrape coverage."""
import json
from pathlib import Path
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[2]

class PrometheusProfileCoverageTests(unittest.TestCase):
    def test_sidecar_monitor_covers_all_scalable_advanced_ticket_services(self):
        script=(ROOT/'experiments/install_prometheus_monitoring.sh').read_text()
        monitor=yaml.safe_load(script.split('<<YAML\n',1)[1].split('\nYAML',1)[0])
        selected=set(monitor['spec']['selector']['matchExpressions'][0]['values'])
        profile=json.loads((ROOT/'ai-module/configs/service_profiles/advanced-ticket.json').read_text())
        required={n['name'] for n in profile['nodes'] if n['scalable']}
        self.assertFalse(required-selected,f'unmonitored services: {sorted(required-selected)}')
