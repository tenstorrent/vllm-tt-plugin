from types import SimpleNamespace

from vllm_tt_plugin import loader


def test_loader_passes_complete_vllm_config_to_model_adapter(monkeypatch):
    received = {}

    class Model:
        @classmethod
        def initialize_vllm_model(cls, *args, **kwargs):
            received.update(kwargs)
            return object()

    config = SimpleNamespace(
        device_config=SimpleNamespace(device="tt"),
        additional_config={},
        parallel_config=SimpleNamespace(data_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=32),
    )
    model_config = SimpleNamespace(
        hf_config=object(),
        max_model_len=131072,
    )
    monkeypatch.setattr(loader, "get_model_architecture", lambda _: (Model, None))

    result = loader.TTModelLoader.load_model(
        object.__new__(loader.TTModelLoader), config, model_config
    )

    assert result is not None
    assert received["vllm_config"] is config

