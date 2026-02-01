# AWS Hosting Cost Analysis

## Executive Summary

This document provides a comprehensive cost analysis for hosting the Artwork Matcher system on AWS, covering scenarios from small museum deployments (10K objects, 1K DAU) to global-scale platforms (1B objects, 10M DAU).

**Key Findings:**
- Small deployments (10K objects, 1K DAU): **~$300-500/month**
- Medium deployments (100K objects, 100K DAU): **~$2,000-4,000/month**
- Large deployments (10M objects, 1M DAU): **~$40,000-60,000/month**
- Global scale (1B objects, 10M DAU): **$1M+/month** (requires custom architecture)

---

## Table of Contents

1. [Assumptions](#assumptions)
2. [Resource Requirements](#resource-requirements)
3. [Scaling Matrices](#scaling-matrices)
4. [Detailed Cost Breakdowns](#detailed-cost-breakdowns)
5. [Architecture Recommendations](#architecture-recommendations)
6. [Optimization Strategies](#optimization-strategies)

---

## Assumptions

### Traffic Patterns

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Requests per DAU | 5 | Museum visitors identify 3-7 artworks per visit |
| Active hours | 12 hours | Museum hours + timezone distribution |
| Peak multiplier | 3× | Lunch and afternoon peaks |
| Peak duration | 4 hours | Concentrated high-traffic window |

### Request Characteristics

| Parameter | Value |
|-----------|-------|
| Average image upload size | 500 KB |
| Average response size | 20 KB |
| Embedding extraction (GPU) | 30-50 ms |
| Embedding extraction (CPU) | 150-300 ms |
| Vector search | <1 ms (up to 1M), <10 ms (up to 100M) |
| Geometric verification | 100-200 ms (5 candidates) |
| **Target total latency** | **<500 ms** |

### AWS Pricing (eu-west-1, On-Demand, 2025)

#### Compute

| Instance | vCPU | RAM | GPU | $/hour | €/hour | $/month |
|----------|------|-----|-----|--------|--------|---------|
| t3.medium | 2 | 4 GB | - | $0.042 | €0.039 | $31 |
| t3.large | 2 | 8 GB | - | $0.083 | €0.076 | $61 |
| c6i.large | 2 | 4 GB | - | $0.085 | €0.078 | $62 |
| c6i.xlarge | 4 | 8 GB | - | $0.170 | €0.156 | $124 |
| c6i.2xlarge | 8 | 16 GB | - | $0.340 | €0.313 | $248 |
| r6i.large | 2 | 16 GB | - | $0.126 | €0.116 | $92 |
| r6i.xlarge | 4 | 32 GB | - | $0.252 | €0.232 | $184 |
| r6i.2xlarge | 8 | 64 GB | - | $0.504 | €0.464 | $368 |
| r6i.4xlarge | 16 | 128 GB | - | $1.008 | €0.927 | $736 |
| r6i.8xlarge | 32 | 256 GB | - | $2.016 | €1.855 | $1,472 |
| r6i.16xlarge | 64 | 512 GB | - | $4.032 | €3.709 | $2,943 |
| **g4dn.xlarge** | 4 | 16 GB | T4 | $0.526 | €0.484 | $384 |
| g5.xlarge | 4 | 16 GB | A10G | $1.006 | €0.926 | $734 |

#### Storage & Network

| Resource | Price |
|----------|-------|
| S3 Standard | $0.023/GB/month |
| EBS gp3 | $0.08/GB/month |
| Data Transfer OUT (first 10TB) | $0.09/GB |
| Data Transfer OUT (10-50TB) | $0.085/GB |
| ALB | $0.0225/hour + $0.008/LCU-hour |
| NAT Gateway | $0.045/hour + $0.045/GB processed |

*EUR prices: multiply USD by 0.92*

---

## Resource Requirements

### Storage Requirements by Index Size

```
Per Object:
├── FAISS embedding:  768 dims × 4 bytes = 3 KB
├── Reference image:  ~500 KB (high-quality JPEG)
├── ORB features:     ~50 KB (pre-computed, optional)
├── Metadata:         ~1 KB
└── Total:            ~554 KB per object
```

| Objects | FAISS Index | Images | Features | Total Storage |
|---------|-------------|--------|----------|---------------|
| **10K** | 30 MB | 5 GB | 0.5 GB | **~6 GB** |
| **100K** | 300 MB | 50 GB | 5 GB | **~56 GB** |
| **1M** | 3 GB | 500 GB | 50 GB | **~554 GB** |
| **10M** | 30 GB | 5 TB | 500 GB | **~5.5 TB** |
| **100M** | 300 GB | 50 TB | 5 TB | **~55 TB** |
| **1B** | 3 TB | 500 TB | 50 TB | **~554 TB** |

### Memory Requirements (FAISS Index)

The FAISS index **must fit in RAM** for fast search:

| Objects | Index Size | Min RAM | Recommended Instance |
|---------|------------|---------|---------------------|
| 10K | 30 MB | 512 MB | t3.medium (4 GB) |
| 100K | 300 MB | 1 GB | t3.medium (4 GB) |
| 1M | 3 GB | 6 GB | r6i.large (16 GB) |
| 10M | 30 GB | 45 GB | r6i.2xlarge (64 GB) |
| 100M | 300 GB | 400 GB | r6i.16xlarge (512 GB) |
| 1B | 3 TB | 4 TB | Sharded/Distributed |

### Service Capacity (QPS per Instance)

| Service | Instance | QPS Capacity | Bottleneck |
|---------|----------|--------------|------------|
| Gateway | t3.medium | 500+ | Network I/O |
| Embeddings (GPU) | g4dn.xlarge | 25-30 | GPU compute |
| Embeddings (CPU) | c6i.xlarge | 3-5 | CPU compute |
| Search | r6i.xlarge | 1,000+ | Memory bandwidth |
| Geometric | c6i.large | 15-20 | CPU compute |

---

## Scaling Matrices

### Matrix 1: Peak QPS Requirements

```
Formula: Peak QPS = (DAU × 5 requests × 3 peak_multiplier) / (4 peak_hours × 3600 sec)
```

| | **1K DAU** | **10K DAU** | **100K DAU** | **1M DAU** | **10M DAU** |
|--|------------|-------------|--------------|------------|-------------|
| **10K obj** | 1 | 10 | 104 | 1,042 | 10,417 |
| **100K obj** | 1 | 10 | 104 | 1,042 | 10,417 |
| **1M obj** | 1 | 10 | 104 | 1,042 | 10,417 |
| **10M obj** | 1 | 10 | 104 | 1,042 | 10,417 |
| **100M obj** | 1 | 10 | 104 | 1,042 | 10,417 |
| **1B obj** | 1 | 10 | 104 | 1,042 | 10,417 |

*QPS independent of index size; depends only on user traffic*

---

### Matrix 2: Inbound Bandwidth (Peak MB/s)

```
Formula: MB/s = Peak_QPS × 0.5 MB (average image size)
```

| | **1K DAU** | **10K DAU** | **100K DAU** | **1M DAU** | **10M DAU** |
|--|------------|-------------|--------------|------------|-------------|
| **10K obj** | 0.5 | 5 | 52 | 521 | 5,208 |
| **100K obj** | 0.5 | 5 | 52 | 521 | 5,208 |
| **1M obj** | 0.5 | 5 | 52 | 521 | 5,208 |
| **10M obj** | 0.5 | 5 | 52 | 521 | 5,208 |
| **100M obj** | 0.5 | 5 | 52 | 521 | 5,208 |
| **1B obj** | 0.5 | 5 | 52 | 521 | 5,208 |

---

### Matrix 3: Instance Count per Service

Format: **Gateway / Embeddings (GPU) / Search / Geometric**

| | **1K DAU** | **10K DAU** | **100K DAU** | **1M DAU** | **10M DAU** |
|--|------------|-------------|--------------|------------|-------------|
| **10K obj** | 1/1/1/1 | 1/1/1/1 | 1/4/1/6 | 3/35/2/53 | 21/348/11/521 |
| **100K obj** | 1/1/1/1 | 1/1/1/1 | 1/4/1/6 | 3/35/2/53 | 21/348/11/521 |
| **1M obj** | 1/1/1/1 | 1/1/1/1 | 1/4/1/6 | 3/35/2/53 | 21/348/11/521 |
| **10M obj** | 1/1/1/1 | 1/1/1/1 | 1/4/2/6 | 3/35/3/53 | 21/348/15/521 |
| **100M obj** | 1/1/1/1 | 1/1/2/1 | 1/4/3/6 | 3/35/5/53 | 21/348/20/521 |
| **1B obj** | 1/1/shd/1 | 1/1/shd/1 | 1/4/shd/6 | 3/35/shd/53 | 21/348/shd/521 |

*shd = sharded/distributed solution required*

**Search Instance Types by Index Size:**
- 10K-100K: t3.medium
- 1M: r6i.large
- 10M: r6i.2xlarge
- 100M: r6i.16xlarge
- 1B: Sharded cluster

---

### Matrix 4: Monthly Cost (USD)

| | **1K DAU** | **10K DAU** | **100K DAU** | **1M DAU** | **10M DAU** |
|--|------------|-------------|--------------|------------|-------------|
| **10K obj** | $520 | $580 | $2,200 | $39,000 | $380,000 |
| **100K obj** | $540 | $600 | $2,250 | $39,500 | $382,000 |
| **1M obj** | $580 | $640 | $2,400 | $40,000 | $385,000 |
| **10M obj** | $850 | $920 | $2,800 | $42,000 | $395,000 |
| **100M obj** | $3,500 | $3,600 | $5,500 | $55,000 | $450,000 |
| **1B obj** | Custom | Custom | Custom | Custom | Custom |

---

### Matrix 5: Monthly Cost (EUR)

*Conversion: USD × 0.92*

| | **1K DAU** | **10K DAU** | **100K DAU** | **1M DAU** | **10M DAU** |
|--|------------|-------------|--------------|------------|-------------|
| **10K obj** | €478 | €534 | €2,024 | €35,880 | €349,600 |
| **100K obj** | €497 | €552 | €2,070 | €36,340 | €351,440 |
| **1M obj** | €534 | €589 | €2,208 | €36,800 | €354,200 |
| **10M obj** | €782 | €846 | €2,576 | €38,640 | €363,400 |
| **100M obj** | €3,220 | €3,312 | €5,060 | €50,600 | €414,000 |
| **1B obj** | Custom | Custom | Custom | Custom | Custom |

---

## Detailed Cost Breakdowns

### Scenario A: Small Museum (10K objects, 10K DAU)

```
┌─────────────────────────────────────────────────────────────────────┐
│              SMALL MUSEUM - 10K objects, 10K DAU                     │
│                     ~$580/month (~€534/month)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  COMPUTE                                                $509  (88%)  │
│  ├── Gateway      1× t3.medium      $31/mo                          │
│  ├── Embeddings   1× g4dn.xlarge    $384/mo  ◀── GPU dominates     │
│  ├── Search       1× t3.medium      $31/mo                          │
│  └── Geometric    1× c6i.large      $62/mo                          │
│                                                                      │
│  STORAGE                                                 $6   (1%)  │
│  ├── S3 images    5 GB              $0.12/mo                        │
│  ├── S3 features  0.5 GB            $0.01/mo                        │
│  └── EBS volumes  4× 20GB gp3       $6.40/mo                        │
│                                                                      │
│  NETWORK                                                $65  (11%)  │
│  ├── ALB          1× ALB            $16/mo                          │
│  ├── Data OUT     ~150 GB           $14/mo                          │
│  └── NAT Gateway  minimal           $35/mo                          │
│                                                                      │
│  ══════════════════════════════════════════════════════════════     │
│  TOTAL                                                  $580/month  │
│                                                         €534/month  │
└─────────────────────────────────────────────────────────────────────┘

Performance:
  • Peak QPS: 10
  • P95 Latency: <400ms
  • Availability: 99% (single AZ)
```

---

### Scenario B: Regional Network (1M objects, 100K DAU)

```
┌─────────────────────────────────────────────────────────────────────┐
│           REGIONAL NETWORK - 1M objects, 100K DAU                    │
│                    ~$2,400/month (~€2,208/month)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  COMPUTE                                              $2,060  (86%)  │
│  ├── Gateway      1× t3.medium      $31/mo                          │
│  ├── Embeddings   4× g4dn.xlarge    $1,536/mo  ◀── 4 GPUs          │
│  ├── Search       1× r6i.large      $92/mo     (16GB for 3GB idx)  │
│  └── Geometric    6× c6i.large      $372/mo                         │
│                                                                      │
│  STORAGE                                                $52   (2%)  │
│  ├── S3 images    500 GB            $12/mo                          │
│  ├── S3 features  50 GB             $1/mo                           │
│  └── EBS volumes  12× 30GB gp3      $29/mo                          │
│                                                                      │
│  NETWORK                                               $288  (12%)  │
│  ├── ALB          1× ALB            $25/mo     (higher LCU)        │
│  ├── Data OUT     ~1.5 TB           $135/mo                         │
│  ├── Cross-AZ     ~200 GB           $4/mo                           │
│  └── NAT Gateway  moderate          $80/mo                          │
│                                                                      │
│  ══════════════════════════════════════════════════════════════     │
│  TOTAL                                                $2,400/month  │
│                                                       €2,208/month  │
└─────────────────────────────────────────────────────────────────────┘

Performance:
  • Peak QPS: 104
  • P95 Latency: <450ms
  • Availability: 99.9% (multi-AZ)
```

---

### Scenario C: National Platform (10M objects, 1M DAU)

```
┌─────────────────────────────────────────────────────────────────────┐
│           NATIONAL PLATFORM - 10M objects, 1M DAU                    │
│                   ~$42,000/month (~€38,640/month)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  COMPUTE                                             $35,800  (85%)  │
│  ├── Gateway      3× c6i.large      $186/mo                         │
│  ├── Embeddings   35× g4dn.xlarge   $13,440/mo  ◀── 35 GPUs        │
│  ├── Search       3× r6i.2xlarge    $1,104/mo  (64GB for 30GB idx) │
│  └── Geometric    53× c6i.large     $3,286/mo                       │
│                                                                      │
│  STORAGE                                               $600   (1%)  │
│  ├── S3 images    5 TB              $115/mo                         │
│  ├── S3 features  500 GB            $12/mo                          │
│  ├── EBS search   3× 100GB gp3      $24/mo                          │
│  └── EBS other    90× 20GB gp3      $144/mo                         │
│                                                                      │
│  NETWORK                                             $3,600   (9%)  │
│  ├── ALB          2× ALB            $80/mo                          │
│  ├── Data OUT     ~15 TB            $1,350/mo                       │
│  ├── Cross-AZ     ~5 TB             $100/mo                         │
│  └── NAT Gateway  high volume       $500/mo                         │
│                                                                      │
│  MANAGED SERVICES                                    $2,000   (5%)  │
│  ├── CloudWatch   logging/metrics   $500/mo                         │
│  ├── ElastiCache  embedding cache   $800/mo   (optional, saves GPU)│
│  ├── Route 53     DNS               $50/mo                          │
│  └── WAF          security          $200/mo                         │
│                                                                      │
│  ══════════════════════════════════════════════════════════════     │
│  TOTAL                                               $42,000/month  │
│                                                      €38,640/month  │
│                                                                      │
│  With Reserved Instances (1-year): ~$29,000/month (-31%)            │
│  With Spot (geometric):            ~$38,000/month (-10%)            │
│  Combined optimizations:           ~$25,000/month (-40%)            │
└─────────────────────────────────────────────────────────────────────┘

Performance:
  • Peak QPS: 1,042
  • P95 Latency: <500ms
  • Availability: 99.95% (multi-AZ, auto-scaling)
```

---

### Scenario D: Global Scale (100M objects, 10M DAU)

```
┌─────────────────────────────────────────────────────────────────────┐
│              GLOBAL SCALE - 100M objects, 10M DAU                    │
│                  ~$450,000/month (~€414,000/month)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  COMPUTE                                            $380,000  (84%)  │
│  ├── Gateway      21× c6i.xlarge    $2,604/mo                       │
│  ├── Embeddings   348× g4dn.xlarge  $133,632/mo  ◀── 348 GPUs!!   │
│  ├── Search       20× r6i.16xlarge  $58,860/mo  (512GB × 20 shards)│
│  └── Geometric    521× c6i.large    $32,302/mo                      │
│                                                                      │
│  STORAGE                                             $5,000   (1%)  │
│  ├── S3 images    50 TB             $1,150/mo   (Intelligent Tier) │
│  ├── S3 features  5 TB              $115/mo                         │
│  └── EBS          massive           $1,500/mo                       │
│                                                                      │
│  NETWORK                                            $35,000   (8%)  │
│  ├── CloudFront   CDN for images    $5,000/mo                       │
│  ├── ALB          10× ALB           $800/mo                         │
│  ├── Data OUT     ~150 TB           $12,750/mo                      │
│  └── Inter-region replication       $5,000/mo                       │
│                                                                      │
│  MANAGED SERVICES                                   $30,000   (7%)  │
│  ├── EKS          cluster mgmt      $2,200/mo                       │
│  ├── ElastiCache  large cluster     $8,000/mo                       │
│  ├── OpenSearch   alternative srch  $10,000/mo  (if replacing FAISS)│
│  └── Operations   monitoring, etc.  $5,000/mo                       │
│                                                                      │
│  ══════════════════════════════════════════════════════════════     │
│  TOTAL (On-Demand)                                 $450,000/month   │
│                                                    €414,000/month   │
│                                                                      │
│  With Full Optimization:           ~$280,000/month (-38%)           │
│  (Reserved + Spot + Caching + Right-sizing)                         │
└─────────────────────────────────────────────────────────────────────┘

Performance:
  • Peak QPS: 10,417
  • P95 Latency: <600ms (geo-distributed)
  • Availability: 99.99% (multi-region)

Architecture Notes:
  • FAISS index sharded across 20 nodes (5M vectors each)
  • Embedding cache reduces GPU load by 40%
  • CloudFront serves reference images
  • Multi-region deployment recommended
```

---

## Architecture Recommendations

### Tier 1: Starter (≤10K objects, ≤10K DAU) — $300-600/mo

```
┌─────────────────────────────────────────────────────────────────┐
│                    STARTER ARCHITECTURE                          │
│                      Single AZ, Minimal HA                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                         ┌─────────┐                              │
│                         │   ALB   │                              │
│                         └────┬────┘                              │
│                              │                                   │
│                         ┌────┴────┐                              │
│                         │ Gateway │                              │
│                         │t3.medium│                              │
│                         └────┬────┘                              │
│              ┌───────────────┼───────────────┐                  │
│              │               │               │                  │
│        ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐           │
│        │Embeddings │   │  Search   │   │ Geometric │           │
│        │g4dn.xlarge│   │ t3.medium │   │ c6i.large │           │
│        │   (GPU)   │   │           │   │           │           │
│        └───────────┘   └───────────┘   └───────────┘           │
│                                                                  │
│   Total: 4 instances                                            │
│   Storage: S3 (images) + EBS (index)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Recommendations:
• Single AZ acceptable (cost savings)
• On-demand instances (flexibility)
• GPU essential for low latency
• Manual scaling sufficient
```

---

### Tier 2: Growth (≤1M objects, ≤100K DAU) — $2,000-5,000/mo

```
┌─────────────────────────────────────────────────────────────────┐
│                     GROWTH ARCHITECTURE                          │
│                      Multi-AZ, Auto-Scaling                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                         ┌─────────┐                              │
│                         │   ALB   │                              │
│                         └────┬────┘                              │
│                              │                                   │
│               ┌──────────────┴──────────────┐                   │
│               │     Gateway ASG (1-3)       │                   │
│               │        t3.medium            │                   │
│               └──────────────┬──────────────┘                   │
│                              │                                   │
│    ┌─────────────────────────┼─────────────────────────┐        │
│    │                         │                         │        │
│    ▼                         ▼                         ▼        │
│ ┌────────────────┐   ┌────────────────┐   ┌────────────────┐   │
│ │ Embeddings ASG │   │  Search (HA)   │   │ Geometric ASG  │   │
│ │    (2-6)       │   │    (1-2)       │   │    (2-10)      │   │
│ │  g4dn.xlarge   │   │   r6i.large    │   │   c6i.large    │   │
│ └────────────────┘   └────────────────┘   └────────────────┘   │
│                                                                  │
│   Total: 6-20 instances (auto-scaling)                          │
│   Storage: S3 + EBS with snapshots                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Recommendations:
• Multi-AZ for 99.9% availability
• Auto Scaling Groups for embeddings/geometric
• Reserved instances for baseline (30% savings)
• Spot instances for geometric burst (70% savings)
• CloudWatch dashboards and alarms
```

---

### Tier 3: Scale (≤100M objects, ≤10M DAU) — $50,000-500,000/mo

```
┌─────────────────────────────────────────────────────────────────┐
│                     SCALE ARCHITECTURE                           │
│                   Kubernetes, Sharded, Global                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────────┐                    ┌─────────────┐           │
│    │ CloudFront  │                    │  Route 53   │           │
│    │    CDN      │                    │   (DNS)     │           │
│    └──────┬──────┘                    └──────┬──────┘           │
│           └─────────────┬────────────────────┘                  │
│                         │                                        │
│                  ┌──────┴──────┐                                │
│                  │ Global ALB  │                                │
│                  └──────┬──────┘                                │
│                         │                                        │
│    ┌────────────────────┴────────────────────┐                  │
│    │           EKS Cluster (Kubernetes)       │                  │
│    │                                          │                  │
│    │  ┌──────────────────────────────────┐   │                  │
│    │  │     Gateway Deployment (HPA)     │   │                  │
│    │  │          10-50 pods              │   │                  │
│    │  └──────────────────────────────────┘   │                  │
│    │                                          │                  │
│    │  ┌──────────────────────────────────┐   │                  │
│    │  │    Embeddings Deployment (HPA)   │   │                  │
│    │  │    GPU Node Pool: 50-500 pods    │   │                  │
│    │  └──────────────────────────────────┘   │                  │
│    │                                          │                  │
│    │  ┌──────────────────────────────────┐   │                  │
│    │  │    Geometric Deployment (HPA)    │   │                  │
│    │  │    Spot Node Pool: 100-600 pods  │   │                  │
│    │  └──────────────────────────────────┘   │                  │
│    │                                          │                  │
│    └──────────────────────────────────────────┘                  │
│                         │                                        │
│    ┌────────────────────┴────────────────────┐                  │
│    │          Search Layer (Sharded)         │                  │
│    │                                          │                  │
│    │  ┌────────┐ ┌────────┐     ┌────────┐  │                  │
│    │  │Shard 1 │ │Shard 2 │ ... │Shard N │  │                  │
│    │  │r6i.16xl│ │r6i.16xl│     │r6i.16xl│  │                  │
│    │  │ 10M ea │ │ 10M ea │     │ 10M ea │  │                  │
│    │  └────────┘ └────────┘     └────────┘  │                  │
│    │                                          │                  │
│    └──────────────────────────────────────────┘                  │
│                                                                  │
│   Additional:                                                    │
│   • ElastiCache cluster for embedding caching                   │
│   • CloudFront for reference images                             │
│   • Multi-region deployment option                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Recommendations:
• Kubernetes (EKS) for orchestration
• Horizontal Pod Autoscaler (HPA) based on QPS
• FAISS sharding: 10-20M vectors per shard
• Reserved + Spot mix (40-50% savings)
• ElastiCache reduces GPU load by 30-50%
• Consider OpenSearch for managed vector search
```

---

## Optimization Strategies

### Reserved Instance Savings

| Commitment | Savings | Best For |
|------------|---------|----------|
| 1-year, no upfront | 30-35% | Uncertain growth |
| 1-year, all upfront | 40-45% | Stable baseline |
| 3-year, all upfront | 55-60% | Predictable workload |

**Example (100K DAU scenario):**
- On-demand: $2,400/month
- 1-year reserved: $1,680/month (-30%)
- 3-year reserved: $1,080/month (-55%)

### Spot Instance Opportunities

| Service | Spot Viable? | Savings | Risk |
|---------|--------------|---------|------|
| Gateway | ⚠️ Careful | 60-70% | Interruption visible to users |
| Embeddings | ⚠️ With fallback | 60-70% | Retry on interruption |
| Search | ❌ No | N/A | Stateful, index in memory |
| Geometric | ✅ Yes | 60-70% | Stateless, easy retry |

### Caching Strategies

| Cache | Hit Rate | Cost | GPU Savings |
|-------|----------|------|-------------|
| ElastiCache (embeddings) | 20-40% | $200-800/mo | 20-40% |
| CloudFront (images) | 80%+ | $50-200/mo | N/A |
| Local ORB features | 100% | Storage only | 40% geometric |

### GPU vs CPU Break-Even

```
GPU (g4dn.xlarge): $0.526/hr → 30 QPS → $0.0049 per 1000 requests
CPU (c6i.xlarge):  $0.170/hr →  4 QPS → $0.0118 per 1000 requests

GPU is 2.4× more cost-effective AND 5× faster.

Conclusion: Always use GPU for embeddings (latency-optimized deployment).
```

---

## This Project's Deployment

For a small museum deployment (20 objects, demo usage):

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEMO DEPLOYMENT OPTIONS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Option A: Local Development (FREE)                              │
│  ├── All services on laptop/desktop                              │
│  ├── CPU inference (~500ms latency)                              │
│  └── Perfect for demo and development                            │
│                                                                  │
│  Option B: Minimal AWS (~$50/month)                              │
│  ├── 1× t3.large (all services combined)                         │
│  ├── CPU inference (~500ms latency)                              │
│  └── Public URL for reviewers                                    │
│                                                                  │
│  Option C: Production-Like (~$400/month)                         │
│  ├── 1× g4dn.xlarge (embeddings)                                 │
│  ├── 1× t3.medium (gateway + search + geometric)                 │
│  ├── GPU inference (~200ms latency)                              │
│  └── Demonstrates real architecture                              │
│                                                                  │
│  Recommendation: Option A for development                        │
│                  Option B/C if deploying live demo               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Scale | Objects | DAU | Monthly Cost (USD) | Monthly Cost (EUR) | Key Infrastructure |
|-------|---------|-----|--------------------|--------------------|-------------------|
| **Starter** | 10K | 1K-10K | $500-600 | €460-550 | 4 instances, single AZ |
| **Growth** | 100K | 100K | $2,000-3,000 | €1,840-2,760 | 10-15 instances, multi-AZ |
| **Scale** | 1M | 1M | $40,000-50,000 | €36,800-46,000 | 50-100 instances, Kubernetes |
| **Enterprise** | 10M | 1M | $40,000-60,000 | €36,800-55,200 | 100+ instances, sharded |
| **Global** | 100M | 10M | $400,000-500,000 | €368,000-460,000 | 500+ instances, multi-region |

**Key Cost Drivers:**
1. **GPU instances (60-70%)** — Embeddings service dominates at all scales
2. **Network egress (5-15%)** — Increases with user traffic
3. **Search memory (5-20%)** — Grows with index size
4. **Geometric compute (10-15%)** — Scales with QPS

**Optimization Potential:** 30-50% reduction with reserved instances, spot, and caching.

---

*Prices based on AWS eu-west-1 (Ireland) region, January 2025. Actual costs may vary. EUR conversions at 0.92 EUR/USD.*
