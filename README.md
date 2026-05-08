# FedSLIP: Federated Sparse LoRA Identity Personalization

**Authors**: Tayyab Waseem, Abdul Hadi

**FedSLIP** is a federated learning framework that achieves **strong zero-shot personalization** under highly non-IID data distributions while maintaining low communication cost. By keeping the sparse LTH mask strictly local, we prevent the client's identity from *slipping* away during global aggregation.


The project uses the AG News dataset with 4 clients exhibiting strong label skew (80% dominant class per client) and a DistilBERT backbone with dual-track LoRA adapters.

## The Core Problem

Standard FedAvg + LoRA suffers from the **Collaboration Penalty**: global averaging dilutes client-specific knowledge. As a result, clients typically require an additional K-step local fine-tuning phase after federation to recover strong personalized performance.

**Baseline Results:**
- FedAvg + LoRA global accuracy: **~90.1%**
- FedAvg + LoRA average client/local accuracy: **~90.4%**
- After extra K-step personalization: **~94%**
- Communication cost: **4.5 MB per round**

## FedSLIP Architecture

FedSLIP introduces a **dual-track LoRA** design:

- **Global Consensus Track**: Shared LoRA adapters trained locally and aggregated on the server via FedAvg + FedProx. Captures general knowledge.
- **Private Sparse Identity Track**: Client-local LoRA module with dynamic Lottery Ticket Hypothesis (LTH) masking. Absorbs user-specific patterns and is never transmitted.

During inference, both tracks are combined, enabling **zero-shot personalization** — strong local accuracy without any post-training adaptation.

## Development Phases

- **Baseline**: Standard FedAvg + LoRA.
- **First Improvement**: Introduced dual-track design. Achieved ~92% personalized accuracy (zero-shot), but suffered from high communication cost (**11.31 MB/round**) and reduced global performance, notably due to client drift.
- **Second Improvement**: Major stabilization with clean optimization separation, FedProx on the global track to reduce client drift, gradient detachment, and private path warmup + freezing.

## Key Results (Phase 2 — 30 rounds)

| Method                        | Global Accuracy   | Personalized Accuracy (Zero-shot) | Comm Cost      |
|-------------------------------|-------------------|-----------------------------------|----------------|
| Baseline FedAvg + LoRA        | 90.1%             | 90.4% (94% after K-step)          | 4.5 MB/round   |
| First Improvement             | Dropped (~74.2%)  | ~92%                              | 11.31 MB/round |
| **Second Improvement**        | **88.0%**         | **~94%**                          | **4.5 MB/round** |

## Notebooks

- `Baseline.ipynb` — Standard FedAvg + LoRA baseline
- `FirstImprovement.ipynb` — Initial dual-track implementation
- `SecondImprovement.ipynb` — Final FedSLIP (Phase 2) with clean decoupling and strong results

## Conclusion

FedSLIP successfully matches the strong personalized performance of the K-step baseline (**~94%**) in a **zero-shot** manner, while keeping communication cost identical to vanilla LoRA (4.5 MB/round). It demonstrates that structural decoupling between global consensus and private identity is a powerful approach for practical personalized federated learning.

