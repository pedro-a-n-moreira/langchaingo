package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	ctx := context.Background()

	// Create OpenAI LLM with reasoning model
	llm, err := openai.New(
		openai.WithModel("o1-mini"), // Reasoning model
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("=== OpenAI Thinking Tokens Demo ===\n")

	// Example 1: Basic reasoning with default settings
	fmt.Println("Example 1: Default Reasoning")
	fmt.Println("-----------------------------")
	
	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Solve this step by step: If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire journey?"),
			},
		},
	}

	resp, err := llm.GenerateContent(ctx, messages,
		llms.WithMaxTokens(500),
	)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", resp.Choices[0].Content)
	printTokenUsage(resp.Choices[0].GenerationInfo)

	// Example 2: Low reasoning effort
	fmt.Println("\nExample 2: Low Reasoning Effort")
	fmt.Println("--------------------------------")
	
	messages2 := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("What is 25 * 4?"),
			},
		},
	}

	resp2, err := llm.GenerateContent(ctx, messages2,
		llms.WithMaxTokens(100),
		llms.WithThinkingMode(llms.ThinkingModeLow),
	)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", resp2.Choices[0].Content)
	printTokenUsage(resp2.Choices[0].GenerationInfo)

	// Example 3: High reasoning effort for complex problem
	fmt.Println("\nExample 3: High Reasoning Effort")
	fmt.Println("---------------------------------")
	
	messages3 := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart(`Consider a modified chess game where:
1. Knights can move like bishops for one turn after capturing a piece
2. Pawns can move backwards one square once per game
3. The king can swap positions with any piece once per game

How would these rule changes affect opening strategy? Analyze the implications.`),
			},
		},
	}

	resp3, err := llm.GenerateContent(ctx, messages3,
		llms.WithMaxTokens(1000),
		llms.WithThinkingMode(llms.ThinkingModeHigh),
	)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", resp3.Choices[0].Content)
	printTokenUsage(resp3.Choices[0].GenerationInfo)

	// Example 4: Streaming with reasoning (if supported)
	fmt.Println("\nExample 4: Streaming with Reasoning")
	fmt.Println("------------------------------------")
	
	messages4 := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Explain the concept of recursion in programming with an example."),
			},
		},
	}

	streamedContent := ""
	streamedReasoning := ""
	
	resp4, err := llm.GenerateContent(ctx, messages4,
		llms.WithMaxTokens(500),
		llms.WithThinkingMode(llms.ThinkingModeMedium),
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			streamedContent += string(chunk)
			fmt.Print(string(chunk))
			return nil
		}),
		llms.WithStreamingReasoningFunc(func(ctx context.Context, reasoningChunk, chunk []byte) error {
			// Capture reasoning tokens (usually hidden in production)
			streamedReasoning += string(reasoningChunk)
			return nil
		}),
	)
	
	fmt.Println() // New line after streaming
	
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	if streamedReasoning != "" {
		fmt.Printf("\n[Note: Model used reasoning tokens during generation]\n")
	}
	
	printTokenUsage(resp4.Choices[0].GenerationInfo)

	// Example 5: Check if model is a reasoning model
	fmt.Println("\nExample 5: Model Detection")
	fmt.Println("---------------------------")
	
	models := []string{
		"o1-preview",
		"o1-mini",
		"o3-mini",
		"gpt-4",
		"gpt-5-mini",
	}
	
	for _, model := range models {
		isReasoning := llms.IsReasoningModel(model)
		fmt.Printf("Model %s: Reasoning=%v\n", model, isReasoning)
	}

	fmt.Println("\n=== Demo Complete ===")
}

func printTokenUsage(generationInfo map[string]any) {
	fmt.Println("\nToken Usage:")
	fmt.Println("------------")
	
	if promptTokens, ok := generationInfo["PromptTokens"].(int); ok {
		fmt.Printf("Prompt Tokens: %d\n", promptTokens)
	}
	
	if completionTokens, ok := generationInfo["CompletionTokens"].(int); ok {
		fmt.Printf("Completion Tokens: %d\n", completionTokens)
	}
	
	if reasoningTokens, ok := generationInfo["ReasoningTokens"].(int); ok && reasoningTokens > 0 {
		fmt.Printf("Reasoning Tokens: %d\n", reasoningTokens)
	}
	
	if totalTokens, ok := generationInfo["TotalTokens"].(int); ok {
		fmt.Printf("Total Tokens: %d\n", totalTokens)
	}
	
	// Extract detailed thinking token usage
	thinkingUsage := llms.ExtractThinkingTokens(generationInfo)
	if thinkingUsage != nil && thinkingUsage.ThinkingTokens > 0 {
		fmt.Println("\nThinking Token Details:")
		fmt.Printf("  Thinking Tokens: %d\n", thinkingUsage.ThinkingTokens)
		if thinkingUsage.ThinkingBudgetAllocated > 0 {
			fmt.Printf("  Budget Allocated: %d\n", thinkingUsage.ThinkingBudgetAllocated)
			fmt.Printf("  Budget Used: %d\n", thinkingUsage.ThinkingBudgetUsed)
			efficiency := float64(thinkingUsage.ThinkingBudgetUsed) / float64(thinkingUsage.ThinkingBudgetAllocated) * 100
			fmt.Printf("  Efficiency: %.1f%%\n", efficiency)
		}
	}
	
	fmt.Println()
}