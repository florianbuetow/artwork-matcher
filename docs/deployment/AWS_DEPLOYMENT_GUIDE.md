# AWS Deployment Guide

## Infrastructure-as-Code Recommendation

**Decision: Terraform**

| Tool | Pros | Cons | Best For |
|------|------|------|----------|
| **Terraform** | Cloud-agnostic, mature ecosystem, excellent state management, large community | HCL learning curve, state file management | Multi-cloud, long-term projects |
| CloudFormation | Native AWS, no state file to manage | AWS-only, verbose YAML/JSON, slower updates | AWS-only shops |
| AWS CDK | Real programming languages, type safety | Abstraction complexity, CloudFormation under hood | Teams preferring TypeScript/Python |
| Pulumi | Real languages, multi-cloud | Smaller community, newer | Polyglot teams |

**Why Terraform for this project:**

1. **Industry standard** — Most widely adopted IaC tool, demonstrates marketable skills
2. **Excellent AWS provider** — Mature, well-documented, covers all services we need
3. **Modular** — Can build reusable modules for each microservice
4. **State management** — Clear visibility into what's deployed
5. **Plan before apply** — Review changes before they happen

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    AWS Cloud                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────────┐   │
│  │   Route 53  │     │                      VPC                             │   │
│  │   (DNS)     │     │  ┌─────────────────────────────────────────────┐    │   │
│  └──────┬──────┘     │  │              Public Subnets                  │    │   │
│         │            │  │  ┌─────────────────────────────────────┐    │    │   │
│         ▼            │  │  │     Application Load Balancer       │    │    │   │
│  ┌─────────────┐     │  │  │         (internet-facing)           │    │    │   │
│  │ CloudFront  │────▶│  │  └──────────────────┬──────────────────┘    │    │   │
│  │   (CDN)     │     │  └────────────────────┼────────────────────────┘    │   │
│  └─────────────┘     │                        │                             │   │
│                      │  ┌─────────────────────┼─────────────────────────┐   │   │
│                      │  │              Private Subnets                   │   │   │
│                      │  │                     │                          │   │   │
│                      │  │    ┌────────────────┴────────────────┐        │   │   │
│                      │  │    │         ECS Cluster              │        │   │   │
│                      │  │    │  ┌─────────┐  ┌─────────┐       │        │   │   │
│                      │  │    │  │ Gateway │  │ Gateway │ ...   │        │   │   │
│                      │  │    │  │ Service │  │ Service │       │        │   │   │
│                      │  │    │  └────┬────┘  └────┬────┘       │        │   │   │
│                      │  │    │       │            │             │        │   │   │
│                      │  │    │  ┌────▼────────────▼────┐       │        │   │   │
│                      │  │    │  │   Internal ALB       │       │        │   │   │
│                      │  │    │  └──────────┬───────────┘       │        │   │   │
│                      │  │    │             │                    │        │   │   │
│                      │  │    │  ┌──────────┼──────────┐        │        │   │   │
│                      │  │    │  │          │          │        │        │   │   │
│                      │  │    │  ▼          ▼          ▼        │        │   │   │
│                      │  │    │ ┌────┐   ┌────┐   ┌────────┐   │        │   │   │
│                      │  │    │ │Emb │   │Srch│   │Geometric│   │        │   │   │
│                      │  │    │ │Svc │   │Svc │   │  Svc   │   │        │   │   │
│                      │  │    │ │GPU │   │    │   │        │   │        │   │   │
│                      │  │    │ └────┘   └──┬─┘   └────────┘   │        │   │   │
│                      │  │    └─────────────┼───────────────────┘        │   │   │
│                      │  │                  │                             │   │   │
│                      │  └──────────────────┼─────────────────────────────┘   │   │
│                      │                     │                                  │   │
│                      └─────────────────────┼──────────────────────────────────┘   │
│                                            │                                      │
│  ┌─────────────────────────────────────────┼────────────────────────────────┐    │
│  │                         Data Layer      │                                 │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───▼───────┐  ┌─────────────┐       │    │
│  │  │     S3      │  │ ElastiCache │  │    EFS    │  │  CloudWatch │       │    │
│  │  │  (images)   │  │  (caching)  │  │  (index)  │  │   (logs)    │       │    │
│  │  └─────────────┘  └─────────────┘  └───────────┘  └─────────────┘       │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Tiers

### Tier 1: Small Demo (~$50-100/month)

Simple deployment for demonstrating the system works.

```
┌─────────────────────────────────────────────┐
│           Single EC2 Instance               │
│  ┌───────────────────────────────────────┐  │
│  │         Docker Compose                 │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐  │  │
│  │  │ Gateway │ │ Search  │ │Geometric│  │  │
│  │  └─────────┘ └─────────┘ └─────────┘  │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │        Embeddings (CPU)         │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Instance: t3.xlarge (4 vCPU, 16 GB)       │
│  Storage: 50 GB gp3                         │
│  Cost: ~$120/month                          │
└─────────────────────────────────────────────┘
```

### Tier 2: Production-Ready (~$500-1,500/month)

Proper microservices with autoscaling and high availability.

```
ECS Fargate (Gateway, Search, Geometric)
+ ECS EC2 with GPU (Embeddings)
+ Application Load Balancer
+ EFS for shared index storage
+ S3 for images
+ CloudWatch for monitoring
```

### Tier 3: Production Scale (~$5,000+/month)

Full production deployment with CDN, caching, multi-AZ.

---

## Project Structure

```
infrastructure/
├── environments/
│   ├── demo/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   │   └── ...
│   └── production/
│       └── ...
├── modules/
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── ecs-cluster/
│   │   └── ...
│   ├── ecs-service/
│   │   └── ...
│   ├── alb/
│   │   └── ...
│   └── autoscaling/
│       └── ...
├── .gitignore
├── backend.tf
└── versions.tf
```

---

## Terraform Configuration

### Backend Configuration (backend.tf)

```hcl
# backend.tf - Remote state storage

terraform {
  backend "s3" {
    bucket         = "artwork-matcher-terraform-state"
    key            = "environments/demo/terraform.tfstate"
    region         = "eu-west-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}
```

### Provider Configuration (versions.tf)

```hcl
# versions.tf

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "artwork-matcher"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
```

---

## Module: VPC (modules/vpc/main.tf)

```hcl
# modules/vpc/main.tf

variable "environment" {
  type        = string
  description = "Environment name (demo, staging, production)"
}

variable "vpc_cidr" {
  type        = string
  default     = "10.0.0.0/16"
  description = "CIDR block for VPC"
}

variable "availability_zones" {
  type        = list(string)
  description = "List of availability zones"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "artwork-matcher-${var.environment}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "artwork-matcher-${var.environment}-igw"
  }
}

# Public Subnets (for ALB)
resource "aws_subnet" "public" {
  count                   = length(var.availability_zones)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "artwork-matcher-${var.environment}-public-${count.index + 1}"
    Type = "public"
  }
}

# Private Subnets (for ECS services)
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index + length(var.availability_zones))
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "artwork-matcher-${var.environment}-private-${count.index + 1}"
    Type = "private"
  }
}

# NAT Gateway (for private subnet internet access)
resource "aws_eip" "nat" {
  count  = var.environment == "demo" ? 1 : length(var.availability_zones)
  domain = "vpc"

  tags = {
    Name = "artwork-matcher-${var.environment}-nat-${count.index + 1}"
  }
}

resource "aws_nat_gateway" "main" {
  count         = var.environment == "demo" ? 1 : length(var.availability_zones)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "artwork-matcher-${var.environment}-nat-${count.index + 1}"
  }

  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "artwork-matcher-${var.environment}-public-rt"
  }
}

resource "aws_route_table" "private" {
  count  = length(var.availability_zones)
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[var.environment == "demo" ? 0 : count.index].id
  }

  tags = {
    Name = "artwork-matcher-${var.environment}-private-rt-${count.index + 1}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Outputs
output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}
```

---

## Module: ECS Cluster (modules/ecs-cluster/main.tf)

```hcl
# modules/ecs-cluster/main.tf

variable "environment" {
  type = string
}

variable "vpc_id" {
  type = string
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "artwork-matcher-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "artwork-matcher-${var.environment}"
  }
}

# Cluster Capacity Providers
resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = "FARGATE"
  }
}

# CloudWatch Log Group for ECS
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/artwork-matcher-${var.environment}"
  retention_in_days = var.environment == "production" ? 30 : 7

  tags = {
    Name = "artwork-matcher-${var.environment}-logs"
  }
}

# Outputs
output "cluster_id" {
  value = aws_ecs_cluster.main.id
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.ecs.name
}
```

---

## Module: ECS Service (modules/ecs-service/main.tf)

```hcl
# modules/ecs-service/main.tf

variable "environment" {
  type = string
}

variable "service_name" {
  type        = string
  description = "Name of the service (gateway, embeddings, search, geometric)"
}

variable "cluster_id" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

variable "container_image" {
  type = string
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "cpu" {
  type        = number
  description = "CPU units (256 = 0.25 vCPU)"
}

variable "memory" {
  type        = number
  description = "Memory in MB"
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "health_check_path" {
  type    = string
  default = "/health"
}

variable "environment_variables" {
  type    = map(string)
  default = {}
}

variable "target_group_arn" {
  type        = string
  default     = null
  description = "ALB target group ARN (optional)"
}

variable "service_registry_arn" {
  type        = string
  default     = null
  description = "Service discovery registry ARN (optional)"
}

variable "log_group_name" {
  type = string
}

variable "enable_gpu" {
  type    = bool
  default = false
}

# Locals
locals {
  full_name = "artwork-matcher-${var.environment}-${var.service_name}"
}

# Security Group
resource "aws_security_group" "service" {
  name        = "${local.full_name}-sg"
  description = "Security group for ${var.service_name} service"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = var.container_port
    to_port     = var.container_port
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
    description = "Allow traffic from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name = "${local.full_name}-sg"
  }
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "task_execution" {
  name = "${local.full_name}-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM Role for ECS Task (application permissions)
resource "aws_iam_role" "task" {
  name = "${local.full_name}-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# S3 access for images and index
resource "aws_iam_role_policy" "task_s3" {
  name = "${local.full_name}-s3-access"
  role = aws_iam_role.task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::artwork-matcher-${var.environment}-*",
          "arn:aws:s3:::artwork-matcher-${var.environment}-*/*"
        ]
      }
    ]
  })
}

# Data sources
data "aws_region" "current" {}

# Task Definition
resource "aws_ecs_task_definition" "service" {
  family                   = local.full_name
  network_mode             = "awsvpc"
  requires_compatibilities = var.enable_gpu ? ["EC2"] : ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = var.service_name
      image     = var.container_image
      essential = true

      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        }
      ]

      environment = [
        for key, value in var.environment_variables : {
          name  = key
          value = value
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = var.log_group_name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = var.service_name
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.container_port}${var.health_check_path} || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      # GPU configuration (for embeddings service)
      resourceRequirements = var.enable_gpu ? [
        {
          type  = "GPU"
          value = "1"
        }
      ] : null
    }
  ])

  tags = {
    Name = local.full_name
  }
}

# ECS Service
resource "aws_ecs_service" "service" {
  name            = local.full_name
  cluster         = var.cluster_id
  task_definition = aws_ecs_task_definition.service.arn
  desired_count   = var.desired_count

  # Use Fargate for non-GPU services
  dynamic "capacity_provider_strategy" {
    for_each = var.enable_gpu ? [] : [1]
    content {
      capacity_provider = "FARGATE"
      weight            = 100
      base              = 1
    }
  }

  # Use EC2 launch type for GPU services
  launch_type = var.enable_gpu ? "EC2" : null

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = false
  }

  # Load balancer configuration (optional)
  dynamic "load_balancer" {
    for_each = var.target_group_arn != null ? [1] : []
    content {
      target_group_arn = var.target_group_arn
      container_name   = var.service_name
      container_port   = var.container_port
    }
  }

  # Enable rolling deployment
  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  # Enable service discovery (for internal service-to-service communication)
  dynamic "service_registries" {
    for_each = var.service_registry_arn != null ? [1] : []
    content {
      registry_arn = var.service_registry_arn
    }
  }

  lifecycle {
    ignore_changes = [desired_count] # Managed by autoscaling
  }

  tags = {
    Name = local.full_name
  }
}

# Outputs
output "service_name" {
  value = aws_ecs_service.service.name
}

output "service_id" {
  value = aws_ecs_service.service.id
}

output "security_group_id" {
  value = aws_security_group.service.id
}

output "task_definition_arn" {
  value = aws_ecs_task_definition.service.arn
}
```

---

## Module: Autoscaling (modules/autoscaling/main.tf)

```hcl
# modules/autoscaling/main.tf

variable "environment" {
  type = string
}

variable "service_name" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "ecs_service_name" {
  type = string
}

variable "min_capacity" {
  type    = number
  default = 1
}

variable "max_capacity" {
  type    = number
  default = 10
}

variable "alb_arn_suffix" {
  type        = string
  default     = ""
  description = "ALB ARN suffix for request-based scaling"
}

variable "scaling_config" {
  type = object({
    cpu_target          = number
    memory_target       = number
    requests_per_target = number
    scale_in_cooldown   = number
    scale_out_cooldown  = number
  })
  default = {
    cpu_target          = 70
    memory_target       = 80
    requests_per_target = 100
    scale_in_cooldown   = 300
    scale_out_cooldown  = 60
  }
}

# Locals
locals {
  resource_id = "service/${var.cluster_name}/${var.ecs_service_name}"
  full_name   = "artwork-matcher-${var.environment}-${var.service_name}"
}

# Application Auto Scaling Target
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = local.resource_id
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# ===== CPU-Based Scaling =====

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${local.full_name}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }

    target_value       = var.scaling_config.cpu_target
    scale_in_cooldown  = var.scaling_config.scale_in_cooldown
    scale_out_cooldown = var.scaling_config.scale_out_cooldown
  }
}

# ===== Memory-Based Scaling =====

resource "aws_appautoscaling_policy" "memory" {
  name               = "${local.full_name}-memory-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }

    target_value       = var.scaling_config.memory_target
    scale_in_cooldown  = var.scaling_config.scale_in_cooldown
    scale_out_cooldown = var.scaling_config.scale_out_cooldown
  }
}

# ===== Request Count Scaling (for ALB-attached services) =====

resource "aws_appautoscaling_policy" "requests" {
  count = var.scaling_config.requests_per_target > 0 && var.alb_arn_suffix != "" ? 1 : 0

  name               = "${local.full_name}-request-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = var.alb_arn_suffix
    }

    target_value       = var.scaling_config.requests_per_target
    scale_in_cooldown  = var.scaling_config.scale_in_cooldown
    scale_out_cooldown = var.scaling_config.scale_out_cooldown
  }
}

# ===== Scheduled Scaling (for predictable traffic patterns) =====

# Scale up before museum opening hours (e.g., 9 AM)
resource "aws_appautoscaling_scheduled_action" "scale_up" {
  count = var.environment == "production" ? 1 : 0

  name               = "${local.full_name}-scale-up-morning"
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  schedule           = "cron(0 8 * * ? *)" # 8 AM UTC daily

  scalable_target_action {
    min_capacity = var.min_capacity * 2
    max_capacity = var.max_capacity
  }
}

# Scale down after closing (e.g., 7 PM)
resource "aws_appautoscaling_scheduled_action" "scale_down" {
  count = var.environment == "production" ? 1 : 0

  name               = "${local.full_name}-scale-down-evening"
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  schedule           = "cron(0 19 * * ? *)" # 7 PM UTC daily

  scalable_target_action {
    min_capacity = var.min_capacity
    max_capacity = var.max_capacity / 2
  }
}

# Outputs
output "autoscaling_target_id" {
  value = aws_appautoscaling_target.ecs.id
}
```

---

## Service-Specific Autoscaling Configurations

Different services have different scaling characteristics:

```hcl
# environments/production/autoscaling.tf

# ===== Gateway Service =====
# Scales based on request count - it's the entry point

module "gateway_autoscaling" {
  source = "../../modules/autoscaling"

  environment      = var.environment
  service_name     = "gateway"
  cluster_name     = module.ecs_cluster.cluster_name
  ecs_service_name = module.gateway_service.service_name

  min_capacity = 2
  max_capacity = 20

  scaling_config = {
    cpu_target          = 60    # Lower threshold - gateway should be responsive
    memory_target       = 70
    requests_per_target = 500   # Scale when each instance gets 500 req/min
    scale_in_cooldown   = 300   # Wait 5 min before scaling in
    scale_out_cooldown  = 60    # Scale out quickly (1 min)
  }
}

# ===== Embeddings Service =====
# Scales based on CPU (GPU utilization) - inference is compute-bound

module "embeddings_autoscaling" {
  source = "../../modules/autoscaling"

  environment      = var.environment
  service_name     = "embeddings"
  cluster_name     = module.ecs_cluster.cluster_name
  ecs_service_name = module.embeddings_service.service_name

  min_capacity = 1
  max_capacity = 10

  scaling_config = {
    cpu_target          = 70    # GPU inference is CPU-bound
    memory_target       = 80
    requests_per_target = 0     # Don't use request-based (internal service)
    scale_in_cooldown   = 600   # GPU instances are expensive, wait 10 min
    scale_out_cooldown  = 120   # GPU startup takes time, 2 min cooldown
  }
}

# ===== Search Service =====
# Scales based on memory - FAISS index is memory-bound

module "search_autoscaling" {
  source = "../../modules/autoscaling"

  environment      = var.environment
  service_name     = "search"
  cluster_name     = module.ecs_cluster.cluster_name
  ecs_service_name = module.search_service.service_name

  min_capacity = 2  # Keep 2 for HA (index is stateful)
  max_capacity = 8

  scaling_config = {
    cpu_target          = 80
    memory_target       = 70    # Lower threshold - FAISS needs headroom
    requests_per_target = 0     # Internal service
    scale_in_cooldown   = 600   # Index loading takes time
    scale_out_cooldown  = 180   # Wait for index to load (3 min)
  }
}

# ===== Geometric Service =====
# Scales based on CPU - ORB/RANSAC is CPU-intensive

module "geometric_autoscaling" {
  source = "../../modules/autoscaling"

  environment      = var.environment
  service_name     = "geometric"
  cluster_name     = module.ecs_cluster.cluster_name
  ecs_service_name = module.geometric_service.service_name

  min_capacity = 1
  max_capacity = 20  # Can scale aggressively - stateless

  scaling_config = {
    cpu_target          = 70    # CPU-intensive operations
    memory_target       = 80
    requests_per_target = 0     # Internal service
    scale_in_cooldown   = 180   # Can scale in faster (stateless)
    scale_out_cooldown  = 30    # Scale out quickly - no startup cost
  }
}
```

---

## Autoscaling Strategy Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Autoscaling Strategy by Service                           │
├─────────────────┬───────────────┬───────────────┬───────────────┬───────────────┤
│                 │    Gateway    │  Embeddings   │    Search     │   Geometric   │
├─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ Primary Metric  │ Request Count │ CPU (GPU)     │ Memory        │ CPU           │
│ Secondary       │ CPU           │ Memory        │ CPU           │ Memory        │
├─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ Min Instances   │ 2             │ 1             │ 2             │ 1             │
│ Max Instances   │ 20            │ 10            │ 8             │ 20            │
├─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ CPU Target      │ 60%           │ 70%           │ 80%           │ 70%           │
│ Memory Target   │ 70%           │ 80%           │ 70%           │ 80%           │
├─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ Scale Out       │ 60s           │ 120s          │ 180s          │ 30s           │
│ Scale In        │ 300s          │ 600s          │ 600s          │ 180s          │
├─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ Spot Eligible   │ ⚠️ Careful    │ ⚠️ Careful    │ ❌ No         │ ✅ Yes        │
│ Reason          │ User-facing   │ GPU cost high │ Stateful      │ Stateless     │
├─────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤
│ Instance Type   │ Fargate       │ EC2 GPU       │ Fargate       │ Fargate       │
│                 │ (256-1024 CPU)│ (g4dn.xlarge) │ (2048+ CPU)   │ (512-2048 CPU)│
└─────────────────┴───────────────┴───────────────┴───────────────┴───────────────┘
```

---

## Main Environment Configuration

### Demo Environment (environments/demo/main.tf)

```hcl
# environments/demo/main.tf

terraform {
  required_version = ">= 1.5.0"

  backend "s3" {
    bucket         = "artwork-matcher-terraform-state"
    key            = "environments/demo/terraform.tfstate"
    region         = "eu-west-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "artwork-matcher"
      Environment = "demo"
      ManagedBy   = "terraform"
    }
  }
}

# Variables
variable "aws_region" {
  default = "eu-west-1"
}

variable "environment" {
  default = "demo"
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# ===== VPC =====

module "vpc" {
  source = "../../modules/vpc"

  environment        = var.environment
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 2)
}

# ===== ECS Cluster =====

module "ecs_cluster" {
  source = "../../modules/ecs-cluster"

  environment = var.environment
  vpc_id      = module.vpc.vpc_id
}

# ===== Application Load Balancer =====

module "alb" {
  source = "../../modules/alb"

  environment       = var.environment
  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnet_ids

  # Health check configuration
  health_check_path = "/health"
}

# ===== ECR Repositories =====

resource "aws_ecr_repository" "services" {
  for_each = toset(["gateway", "embeddings", "search", "geometric"])

  name                 = "artwork-matcher/${each.key}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ===== Services =====

# Gateway Service
module "gateway_service" {
  source = "../../modules/ecs-service"

  environment     = var.environment
  service_name    = "gateway"
  cluster_id      = module.ecs_cluster.cluster_id
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids
  container_image = "${aws_ecr_repository.services["gateway"].repository_url}:latest"
  container_port  = 8000
  cpu             = 256
  memory          = 512
  desired_count   = 1
  log_group_name  = module.ecs_cluster.log_group_name
  target_group_arn = module.alb.target_group_arn

  environment_variables = {
    GATEWAY__BACKENDS__EMBEDDINGS__URL = "http://embeddings.artwork-matcher.local:8000"
    GATEWAY__BACKENDS__SEARCH__URL     = "http://search.artwork-matcher.local:8000"
    GATEWAY__BACKENDS__GEOMETRIC__URL  = "http://geometric.artwork-matcher.local:8000"
  }
}

# Embeddings Service (CPU for demo, GPU for production)
module "embeddings_service" {
  source = "../../modules/ecs-service"

  environment     = var.environment
  service_name    = "embeddings"
  cluster_id      = module.ecs_cluster.cluster_id
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids
  container_image = "${aws_ecr_repository.services["embeddings"].repository_url}:latest"
  container_port  = 8000
  cpu             = 1024   # 1 vCPU for demo
  memory          = 4096   # 4 GB for model
  desired_count   = 1
  log_group_name  = module.ecs_cluster.log_group_name
  enable_gpu      = false  # CPU for demo (set true for production)

  environment_variables = {
    EMBEDDINGS__MODEL__DEVICE = "cpu"
  }
}

# Search Service
module "search_service" {
  source = "../../modules/ecs-service"

  environment     = var.environment
  service_name    = "search"
  cluster_id      = module.ecs_cluster.cluster_id
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids
  container_image = "${aws_ecr_repository.services["search"].repository_url}:latest"
  container_port  = 8000
  cpu             = 512
  memory          = 1024
  desired_count   = 1
  log_group_name  = module.ecs_cluster.log_group_name

  environment_variables = {
    SEARCH__INDEX__PATH = "/data/index/faiss.index"
  }
}

# Geometric Service
module "geometric_service" {
  source = "../../modules/ecs-service"

  environment     = var.environment
  service_name    = "geometric"
  cluster_id      = module.ecs_cluster.cluster_id
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids
  container_image = "${aws_ecr_repository.services["geometric"].repository_url}:latest"
  container_port  = 8000
  cpu             = 512
  memory          = 1024
  desired_count   = 1
  log_group_name  = module.ecs_cluster.log_group_name
}

# ===== S3 Buckets =====

resource "aws_s3_bucket" "data" {
  bucket = "artwork-matcher-${var.environment}-data"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ===== Outputs =====

output "alb_dns_name" {
  value       = module.alb.dns_name
  description = "DNS name of the load balancer"
}

output "ecr_repositories" {
  value = {
    for name, repo in aws_ecr_repository.services :
    name => repo.repository_url
  }
  description = "ECR repository URLs"
}

output "s3_bucket" {
  value       = aws_s3_bucket.data.id
  description = "S3 bucket for data"
}
```

---

## Deployment Commands

### Initial Setup

```bash
# 1. Create S3 bucket for Terraform state
aws s3 mb s3://artwork-matcher-terraform-state --region eu-west-1

# 2. Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region eu-west-1

# 3. Initialize Terraform
cd infrastructure/environments/demo
terraform init

# 4. Review the plan
terraform plan -out=tfplan

# 5. Apply
terraform apply tfplan
```

### Build and Push Docker Images

```bash
# Login to ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-1.amazonaws.com

# Build and push each service
for service in gateway embeddings search geometric; do
  docker build -t artwork-matcher/$service:latest ./services/$service
  docker tag artwork-matcher/$service:latest \
    $(terraform output -raw ecr_repositories | jq -r ".$service"):latest
  docker push $(terraform output -raw ecr_repositories | jq -r ".$service"):latest
done

# Force ECS to pull new images
aws ecs update-service \
  --cluster artwork-matcher-demo \
  --service artwork-matcher-demo-gateway \
  --force-new-deployment
```

### Scaling Commands

```bash
# Manual scale (temporary)
aws ecs update-service \
  --cluster artwork-matcher-demo \
  --service artwork-matcher-demo-gateway \
  --desired-count 3

# Update autoscaling limits
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/artwork-matcher-demo/artwork-matcher-demo-gateway \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10
```

---

## Monitoring & Alerts

```hcl
# modules/monitoring/main.tf

variable "environment" {
  type = string
}

variable "aws_region" {
  type = string
}

variable "alb_arn_suffix" {
  type = string
}

variable "sns_topic_arn" {
  type = string
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "artwork-matcher-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Request Latency (P99)"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", var.alb_arn_suffix,
              { stat = "p99", period = 60, label = "P99 Latency" }]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Request Count"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", var.alb_arn_suffix,
              { stat = "Sum", period = 60 }]
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 24
        height = 6
        properties = {
          title  = "ECS Service CPU Utilization"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ServiceName", "artwork-matcher-${var.environment}-gateway",
              "ClusterName", "artwork-matcher-${var.environment}",
              { stat = "Average", period = 60, label = "gateway" }],
            ["AWS/ECS", "CPUUtilization", "ServiceName", "artwork-matcher-${var.environment}-embeddings",
              "ClusterName", "artwork-matcher-${var.environment}",
              { stat = "Average", period = 60, label = "embeddings" }],
            ["AWS/ECS", "CPUUtilization", "ServiceName", "artwork-matcher-${var.environment}-search",
              "ClusterName", "artwork-matcher-${var.environment}",
              { stat = "Average", period = 60, label = "search" }],
            ["AWS/ECS", "CPUUtilization", "ServiceName", "artwork-matcher-${var.environment}-geometric",
              "ClusterName", "artwork-matcher-${var.environment}",
              { stat = "Average", period = 60, label = "geometric" }]
          ]
        }
      }
    ]
  })
}

# High Latency Alert
resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "artwork-matcher-${var.environment}-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "p99"
  threshold           = 2.0  # 2 seconds
  alarm_description   = "Response latency is above 2 seconds"
  alarm_actions       = [var.sns_topic_arn]

  dimensions = {
    LoadBalancer = var.alb_arn_suffix
  }
}

# Error Rate Alert
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "artwork-matcher-${var.environment}-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 5  # 5% error rate

  metric_query {
    id          = "error_rate"
    expression  = "(errors / requests) * 100"
    label       = "Error Rate"
    return_data = true
  }

  metric_query {
    id = "errors"
    metric {
      metric_name = "HTTPCode_Target_5XX_Count"
      namespace   = "AWS/ApplicationELB"
      period      = 60
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
      }
    }
  }

  metric_query {
    id = "requests"
    metric {
      metric_name = "RequestCount"
      namespace   = "AWS/ApplicationELB"
      period      = 60
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
      }
    }
  }

  alarm_actions = [var.sns_topic_arn]
}
```

---

## Cost Optimization Tips

### 1. Use Fargate Spot for Geometric Service

```hcl
# Geometric is stateless - perfect for Spot

resource "aws_ecs_service" "geometric" {
  # ...

  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 80  # 80% Spot
    base              = 0
  }

  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 20  # 20% On-Demand for baseline
    base              = 1   # At least 1 on-demand
  }
}
```

### 2. Reserved Capacity for Predictable Workloads

```hcl
# Use Savings Plans or Reserved Instances for:
# - Gateway (always running)
# - Search (always running, holds index)
# - Embeddings (if consistent usage)

# Calculate: (monthly_hours * on_demand_rate) vs reserved_rate
# 720 hours * $0.04/hour = $28.80/month on-demand
# Reserved: ~$20/month (30% savings)
```

### 3. Right-size Based on Metrics

```bash
# Check actual resource usage
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=artwork-matcher-demo-gateway \
               Name=ClusterName,Value=artwork-matcher-demo \
  --start-time $(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 3600 \
  --statistics Average Maximum

# If CPU avg < 30% and max < 60%, consider downsizing
```

---

## Summary: Deployment Checklist

### For Small Demo

```
□ Create AWS account (if needed)
□ Set up Terraform state bucket
□ Deploy VPC and ECS cluster
□ Build and push Docker images
□ Deploy services (CPU-only for cost savings)
□ Verify /health endpoints
□ Test /identify endpoint
□ Document the deployed URL
```

### For Production

```
□ Multi-AZ deployment
□ GPU instances for embeddings
□ Autoscaling configured per service
□ CloudWatch alarms set up
□ SSL certificate on ALB
□ WAF for API protection
□ Backup strategy for S3/EFS
□ CI/CD pipeline (GitHub Actions / GitLab CI)
□ Cost monitoring with budgets
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Initialize | `terraform init` |
| Plan | `terraform plan -out=tfplan` |
| Apply | `terraform apply tfplan` |
| Destroy | `terraform destroy` |
| Push images | `docker push <ecr-url>:latest` |
| Force redeploy | `aws ecs update-service --force-new-deployment` |
| Scale manually | `aws ecs update-service --desired-count N` |
| View logs | `aws logs tail /ecs/artwork-matcher-demo --follow` |
| Check health | `curl https://<alb-dns>/health` |
