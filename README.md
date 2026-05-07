# FedSLIP

We introduce FedSLIP: Sparse Local Identity Personalization. By keeping the sparse LTH mask strictly local, we prevent the client's identity from slipping away during global aggregation.

### Abstract: 

Current federated learning forces a compromise between collaborative knowledge and personal identity. We built a dual-track architecture that isolates a user's specific data skew inside a highly sparse, structurally private LTH mask. By decoupling the global consensus from the local identity, our architecture achieves zero-shot personalization natively during the federated process, strictly preventing model inversion privacy leaks, without adding a single byte of communication overhead to the network payload.
