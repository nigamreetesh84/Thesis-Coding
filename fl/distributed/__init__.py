"""Distributed deployment of the federated rule extractor (RQ5, round 8).

Run 1 simulated FedAvg in one process. This package runs the SAME training code
(fl.train) and the SAME FedAvg rule as a real distributed system built on the
Flower framework (Deployment Engine):

* SuperLink  -- the aggregation server; runs ``server_app`` (FedAvg over LoRA adapters)
* SuperNode  -- one long-running process per bank (SBI, HDFC, AXIS); runs
                ``client_app`` next to that bank's own data folder
* ``flwr run`` -- submits an experiment to the SuperLink

Only LoRA adapter weights (Flower ``ArrayRecord``) and a few scalar metrics travel
over gRPC; each bank's training file never leaves its node.

Modules:
    codec       LoRA state dict <-> ArrayRecord, byte counting
    settings    Flower run config <-> FLConfig
    prepare     per-bank data folders + server test-set folder (+ manifest)
    client_app  Flower ClientApp (bank side)
    server_app  Flower ServerApp (aggregation + final evaluation)
    compare     distributed run vs simulated run (same settings)
    tiny_model  offline tiny model for smoke tests (no download)

See DISTRIBUTED_FL.md for the commands.
"""
