import json
from types import SimpleNamespace
import unittest

from benchmark_platform.harnesses.api import ApiConfig
from benchmark_platform.harnesses.artifact_provenance import provider_identity


class ArtifactProvenanceTests(unittest.TestCase):
    def test_model_configuration_is_recorded_without_credentials_or_endpoint(self):
        config = ApiConfig(base_url="https://private.example/v1", api_key="synthetic-secret", model="test-model",
                           reasoning_effort="low", stream=True)
        identity = provider_identity(SimpleNamespace(config=config))
        self.assertEqual(identity["model"], "test-model")
        self.assertEqual(identity["reasoning_effort"], "low")
        self.assertTrue(identity["stream"])
        self.assertEqual(len(identity["endpoint_sha256"]), 64)
        self.assertNotIn("synthetic-secret", json.dumps(identity))
        self.assertNotIn("private.example", json.dumps(identity))
