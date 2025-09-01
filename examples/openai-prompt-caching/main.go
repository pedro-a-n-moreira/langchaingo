package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	ctx := context.Background()

	// Create OpenAI LLM - GPT-4o supports automatic prompt caching
	llm, err := openai.New(
		openai.WithModel("gpt-4o"), // Supports automatic prompt caching
	)
	if err != nil {
		log.Fatal(err)
	}

	// OpenAI's prompt caching is automatic for prompts >1024 tokens
	// Create a long system prompt that will trigger automatic caching
	systemPrompt := createLongSystemPrompt()
	
	fmt.Printf("System prompt length: %d characters (>1024 tokens should trigger caching)\n\n", len(systemPrompt))
	
	// Example 1: First request with long prompt
	fmt.Println("=== Example 1: First Request (Cache Miss) ===")
	
	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeSystem,
			Parts: []llms.ContentPart{
				llms.TextPart(systemPrompt),
			},
		},
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("What are the key principles of microservices architecture?"),
			},
		},
	}

	start := time.Now()
	resp, err := llm.GenerateContent(ctx, messages,
		llms.WithMaxTokens(200),
		llms.WithTemperature(0.1), // Low temperature for consistent responses
	)
	duration := time.Now().Sub(start)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("First request took: %v\n", duration)
	fmt.Printf("Response: %s\n\n", resp.Choices[0].Content)

	// Example 2: Second request with the same prompt prefix (should hit cache)
	fmt.Println("=== Example 2: Second Request (Cache Hit Expected) ===")
	
	// Same system prompt, different user question
	messages[1] = llms.MessageContent{
		Role: llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{
			llms.TextPart("How do you handle data consistency in microservices?"),
		},
	}

	start = time.Now()
	resp2, err := llm.GenerateContent(ctx, messages,
		llms.WithMaxTokens(200),
		llms.WithTemperature(0.1),
	)
	duration2 := time.Now().Sub(start)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Second request took: %v (should be faster due to automatic caching)\n", duration2)
	fmt.Printf("Response: %s\n\n", resp2.Choices[0].Content)

	// Example 3: Different prompt (cache miss)
	fmt.Println("=== Example 3: Different Prompt (Cache Miss) ===")
	
	differentMessages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeSystem,
			Parts: []llms.ContentPart{
				llms.TextPart("You are a helpful AI assistant specializing in database design."),
			},
		},
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("What are database normalization best practices?"),
			},
		},
	}

	start = time.Now()
	resp3, err := llm.GenerateContent(ctx, differentMessages,
		llms.WithMaxTokens(200),
		llms.WithTemperature(0.1),
	)
	duration3 := time.Now().Sub(start)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Different prompt request took: %v (new cache entry)\n", duration3)
	fmt.Printf("Response: %s\n\n", resp3.Choices[0].Content)

	// Example 4: Enable explicit prompt caching awareness (metadata)
	fmt.Println("=== Example 4: With Prompt Caching Metadata ===")
	
	resp4, err := llm.GenerateContent(ctx, messages,
		llms.WithMaxTokens(150),
		llms.WithTemperature(0.1),
		llms.WithPromptCaching(true), // Explicit caching metadata
	)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("With caching metadata response: %s\n", resp4.Choices[0].Content)

	fmt.Println("\n=== OpenAI Automatic Prompt Caching Demo Complete ===")
	fmt.Println("Key points about OpenAI prompt caching:")
	fmt.Printf("- Automatic for prompts >1024 tokens\n")
	fmt.Printf("- 50%% discount on cached input tokens\n")
	fmt.Printf("- Up to 80%% latency reduction\n")
	fmt.Printf("- Cache duration: 5-10 minutes of inactivity\n")
	fmt.Printf("- No code changes required\n")
}

func createLongSystemPrompt() string {
	// Create a system prompt that's definitely >1024 tokens to trigger caching
	base := `You are an expert software architect and consultant specializing in distributed systems and microservices.

Your expertise includes:

MICROSERVICES ARCHITECTURE:
- Service decomposition strategies and domain-driven design
- API gateway patterns and service mesh implementation
- Inter-service communication (synchronous vs asynchronous)
- Event-driven architectures and message queuing systems
- Service discovery and load balancing strategies
- Circuit breaker patterns and fault tolerance
- Distributed tracing and observability
- Container orchestration with Kubernetes and Docker

DATABASE AND DATA MANAGEMENT:
- Database per service pattern and data consistency challenges
- CQRS (Command Query Responsibility Segregation) and Event Sourcing
- Polyglot persistence and choosing the right database
- Distributed transaction patterns (Saga, Two-Phase Commit)
- Data synchronization and eventual consistency
- Caching strategies (Redis, Memcached, distributed caching)
- Database sharding and replication patterns

SCALABILITY AND PERFORMANCE:
- Horizontal vs vertical scaling strategies
- Auto-scaling and resource optimization
- Performance monitoring and APM tools
- Load testing strategies and capacity planning
- CDN integration and edge computing
- Asynchronous processing and job queues

SECURITY AND COMPLIANCE:
- OAuth 2.0 and JWT token management
- API security best practices and rate limiting
- Service-to-service authentication and authorization
- Secrets management and configuration security
- Compliance frameworks (SOC 2, GDPR, HIPAA)
- Security scanning and vulnerability management

DEVOPS AND DEPLOYMENT:
- CI/CD pipeline design and implementation
- Infrastructure as Code (Terraform, CloudFormation)
- Blue-green and canary deployment strategies
- Monitoring and alerting systems (Prometheus, Grafana)
- Log aggregation and analysis (ELK Stack, Splunk)
- Disaster recovery and backup strategies

CLOUD PLATFORMS:
- AWS, Azure, and Google Cloud services
- Serverless architectures and Function-as-a-Service
- Cloud-native patterns and 12-factor app principles
- Cost optimization and resource management
- Multi-cloud and hybrid cloud strategies

Please provide detailed, practical advice based on industry best practices and real-world experience.`

	// Repeat parts to ensure we're well over 1024 tokens
	repeated := strings.Repeat("When answering questions, consider scalability, maintainability, security, and performance implications. Provide specific examples and code snippets when helpful. ", 10)
	
	return base + "\n\n" + repeated + "\n\nAlways explain the reasoning behind your recommendations and mention potential trade-offs."
}