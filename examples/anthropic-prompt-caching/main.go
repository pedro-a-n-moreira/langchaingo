package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"
)

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		log.Fatal("ANTHROPIC_API_KEY environment variable is required")
	}

	ctx := context.Background()

	// Create Anthropic LLM
	llm, err := anthropic.New(
		anthropic.WithModel("claude-3-haiku-20240307"), // Supports prompt caching
	)
	if err != nil {
		log.Fatal(err)
	}

	// Example 1: Basic prompt caching with system message
	fmt.Println("=== Example 1: Basic Prompt Caching ===")
	
	// Long system prompt that we want to cache
	systemPrompt := `You are Claude, an AI assistant created by Anthropic. You are helpful, harmless, and honest.
	You have extensive knowledge about software engineering, mathematics, science, literature, and many other topics.
	You can help with coding problems, explain complex concepts, assist with writing, and engage in thoughtful conversation.
	Always strive to be accurate and acknowledge when you're uncertain about something.
	When helping with code, provide clear explanations and follow best practices for the programming language being used.
	Be concise but thorough in your explanations, and ask clarifying questions when needed.`

	// Wrap the system prompt with cache control
	cachedSystemPart := llms.WithCacheControl(
		llms.TextPart(systemPrompt),
		llms.EphemeralCache(), // 5-minute cache
	)

	messages := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeSystem,
			Parts: []llms.ContentPart{cachedSystemPart},
		},
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("What are the key principles of clean code?"),
			},
		},
	}

	start := time.Now()
	resp, err := llm.GenerateContent(ctx, messages,
		llms.WithAnthropicCachingHeaders(), // Enable prompt caching
		llms.WithMaxTokens(200),
	)
	duration := time.Since(start)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("First request took: %v\n", duration)
	fmt.Printf("Response: %s\n\n", resp.Choices[0].Content)

	// Example 2: Make the same request again to demonstrate cache hit
	fmt.Println("=== Example 2: Cache Hit (Same System Prompt) ===")
	
	messages[1] = llms.MessageContent{
		Role: llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{
			llms.TextPart("Give me one specific example of clean code practice."),
		},
	}

	start = time.Now()
	resp2, err := llm.GenerateContent(ctx, messages,
		llms.WithAnthropicCachingHeaders(),
		llms.WithMaxTokens(150),
	)
	duration2 := time.Since(start)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Second request took: %v (should be faster due to caching)\n", duration2)
	fmt.Printf("Response: %s\n\n", resp2.Choices[0].Content)

	// Example 3: Long-lived cache (1 hour)
	fmt.Println("=== Example 3: One-Hour Cache ===")
	
	longContext := `Context: You are analyzing a large codebase with the following structure and patterns:
	
	The project follows a microservices architecture with the following services:
	- User Service: Handles authentication, user profiles, and user management
	- Product Service: Manages product catalog, inventory, and pricing
	- Order Service: Processes orders, payments, and fulfillment
	- Notification Service: Sends emails, SMS, and push notifications
	
	Technology Stack:
	- Backend: Go with Gin framework
	- Database: PostgreSQL with Redis for caching
	- Message Queue: RabbitMQ for async processing
	- Container Orchestration: Kubernetes
	- Monitoring: Prometheus + Grafana
	- CI/CD: GitHub Actions
	
	Code Standards:
	- All services follow Clean Architecture principles
	- Dependency injection using wire
	- Error handling with structured logging
	- Comprehensive test coverage (unit, integration, e2e)
	- API documentation with OpenAPI 3.0
	
	Current Focus: We're working on improving the performance and reliability of the Order Service.`

	longCachedPart := llms.WithCacheControl(
		llms.TextPart(longContext),
		llms.EphemeralCacheOneHour(), // 1-hour cache
	)

	longMessages := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeSystem,
			Parts: []llms.ContentPart{longCachedPart},
		},
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("What are the main performance optimization strategies I should consider for the Order Service?"),
			},
		},
	}

	start = time.Now()
	resp3, err := llm.GenerateContent(ctx, longMessages,
		llms.WithAnthropicCachingHeaders(),
		llms.WithMaxTokens(300),
	)
	duration3 := time.Since(start)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Long context request took: %v\n", duration3)
	fmt.Printf("Response: %s\n\n", resp3.Choices[0].Content)

	// Example 4: Mixed cached and non-cached content
	fmt.Println("=== Example 4: Mixed Content ===")
	
	mixedMessages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				// Cached part
				llms.WithCacheControl(
					llms.TextPart("Here's a large document to analyze:\n\n"+longContext),
					llms.EphemeralCache(),
				),
				// Non-cached part
				llms.TextPart("\n\nSpecific question: What database indexing strategies would you recommend?"),
			},
		},
	}

	resp4, err := llm.GenerateContent(ctx, mixedMessages,
		llms.WithAnthropicCachingHeaders(),
		llms.WithMaxTokens(250),
	)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Mixed content response: %s\n", resp4.Choices[0].Content)
	
	fmt.Println("\n=== Prompt Caching Demo Complete ===")
	fmt.Println("Benefits observed:")
	fmt.Printf("- Reduced latency for cached prompts\n")
	fmt.Printf("- Cost savings (50%% discount on cached tokens)\n")
	fmt.Printf("- Improved user experience for repeated contexts\n")
}