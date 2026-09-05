"""Credential-free model configuration recorded by offline optimizers."""
import hashlib


def provider_identity(client) -> dict:
    config = getattr(client, "config", None)
    if config is None:
        return {"kind": "unreported-client-configuration"}
    result = {key: getattr(config, key, None) for key in (
        "model", "api_type", "reasoning_effort", "stream", "temperature", "max_output_tokens",
        "timeout_seconds", "transport_retries",
    )}
    result["endpoint_sha256"] = hashlib.sha256(config.base_url.encode()).hexdigest()
    return result
