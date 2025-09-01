package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/llms/openai"
)

// TestConfig represents a test configuration
type TestConfig struct {
	Name           string
	ThinkingMode   llms.ThinkingMode
	Streaming      bool
	Model          string // Override model if needed
}

// Backend represents an LLM backend
type Backend struct {
	Name           string
	LLM            llms.Model
	DefaultModel   string
	ReasoningModel string // Model that supports reasoning
	Enabled        bool
}

func main() {
	ctx := context.Background()

	fmt.Println("=== Comprehensive Multi-Backend Thinking Tokens Demo ===")
	fmt.Println("Testing: Thinking vs Non-Thinking | Streaming vs Non-Streaming")
	fmt.Println("=" + strings.Repeat("=", 55) + "\n")

	// Initialize backends
	backends := initializeBackends()

	// Test configurations - all combinations
	configs := []TestConfig{
		// Non-thinking modes
		{Name: "Standard (No Thinking, No Streaming)", ThinkingMode: llms.ThinkingModeNone, Streaming: false},
		{Name: "Standard + Streaming", ThinkingMode: llms.ThinkingModeNone, Streaming: true},
		
		// Thinking modes without streaming
		{Name: "Low Thinking", ThinkingMode: llms.ThinkingModeLow, Streaming: false},
		{Name: "Medium Thinking", ThinkingMode: llms.ThinkingModeMedium, Streaming: false},
		{Name: "High Thinking", ThinkingMode: llms.ThinkingModeHigh, Streaming: false},
		
		// Thinking modes with streaming
		{Name: "Low Thinking + Streaming", ThinkingMode: llms.ThinkingModeLow, Streaming: true},
		{Name: "Medium Thinking + Streaming", ThinkingMode: llms.ThinkingModeMedium, Streaming: true},
		{Name: "High Thinking + Streaming", ThinkingMode: llms.ThinkingModeHigh, Streaming: true},
	}

	// Test prompt - something that benefits from reasoning
	prompt := "What is the next number in the sequence: 2, 6, 12, 20, 30, ?"

	// Run tests for each backend
	for _, backend := range backends {
		if !backend.Enabled {
			fmt.Printf("\n[%s Backend - SKIPPED]\n", backend.Name)
			fmt.Printf("Reason: API key not configured (%s_API_KEY)\n", getEnvVarName(backend.Name))
			continue
		}

		fmt.Printf("\n%s\n", strings.Repeat("=", 60))
		fmt.Printf("BACKEND: %s\n", backend.Name)
		fmt.Printf("Default Model: %s\n", backend.DefaultModel)
		fmt.Printf("Reasoning Model: %s\n", backend.ReasoningModel)
		
		// Check if backend supports reasoning
		if reasoner, ok := backend.LLM.(llms.ReasoningModel); ok {
			fmt.Printf("Supports Reasoning: %v\n", reasoner.SupportsReasoning())
		} else {
			fmt.Printf("Supports Reasoning: Interface not implemented\n")
		}
		fmt.Printf("%s\n", strings.Repeat("=", 60))

		for _, config := range configs {
			// Skip thinking modes if backend doesn't have a reasoning model
			if config.ThinkingMode != llms.ThinkingModeNone && backend.ReasoningModel == "" {
				fmt.Printf("\n[%s] - SKIPPED (no reasoning model available)\n", config.Name)
				continue
			}

			fmt.Printf("\n[%s]\n", config.Name)
			fmt.Println(strings.Repeat("-", len(config.Name)+2))

			// Determine which model to use
			model := backend.DefaultModel
			if config.ThinkingMode != llms.ThinkingModeNone && backend.ReasoningModel != "" {
				model = backend.ReasoningModel
			}

			// Run the test
			testBackendConfig(ctx, backend.LLM, model, prompt, config)
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("Demo Complete!")
}

func initializeBackends() []Backend {
	var backends []Backend

	// OpenAI Backend
	if os.Getenv("OPENAI_API_KEY") != "" {
		llm, err := openai.New(
			openai.WithModel("gpt-4-turbo-preview"),
		)
		if err == nil {
			backends = append(backends, Backend{
				Name:           "OpenAI",
				LLM:            llm,
				DefaultModel:   "gpt-4-turbo-preview",
				ReasoningModel: "o1-mini", // Use o1-mini for reasoning tests
				Enabled:        true,
			})
		}
	} else {
		backends = append(backends, Backend{
			Name:    "OpenAI",
			Enabled: false,
		})
	}

	// Anthropic Backend
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		llm, err := anthropic.New(
			anthropic.WithModel("claude-3-sonnet-20240229"),
		)
		if err == nil {
			backends = append(backends, Backend{
				Name:           "Anthropic",
				LLM:            llm,
				DefaultModel:   "claude-3-sonnet-20240229",
				ReasoningModel: "claude-3-7-sonnet", // Use Claude 3.7 for reasoning
				Enabled:        true,
			})
		}
	} else {
		backends = append(backends, Backend{
			Name:    "Anthropic",
			Enabled: false,
		})
	}

	// Ollama Backend (local)
	llm, err := ollama.New(
		ollama.WithModel("llama2"),
	)
	if err == nil {
		// Test if Ollama is actually running
		testMsg := []llms.MessageContent{{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart("test")},
		}}
		_, testErr := llm.GenerateContent(context.Background(), testMsg,
			llms.WithMaxTokens(1),
		)
		
		if testErr == nil || !strings.Contains(testErr.Error(), "connection refused") {
			// Check for available models
			reasoningModel := ""
			// Try to detect DeepSeek R1 or other reasoning models
			// This would normally involve checking available models via Ollama API
			if checkOllamaModel("deepseek-r1") {
				reasoningModel = "deepseek-r1"
			} else if checkOllamaModel("qwen2.5-coder:32b-instruct") {
				reasoningModel = "qwen2.5-coder:32b-instruct"
			}
			
			backends = append(backends, Backend{
				Name:           "Ollama",
				LLM:            llm,
				DefaultModel:   "llama2",
				ReasoningModel: reasoningModel,
				Enabled:        true,
			})
		} else {
			backends = append(backends, Backend{
				Name:    "Ollama",
				Enabled: false,
			})
		}
	} else {
		backends = append(backends, Backend{
			Name:    "Ollama",
			Enabled: false,
		})
	}

	return backends
}

func testBackendConfig(ctx context.Context, llm llms.Model, model string, prompt string, config TestConfig) {
	messages := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart(prompt)},
		},
	}

	// Build options
	opts := []llms.CallOption{
		llms.WithModel(model),
		llms.WithMaxTokens(200),
		llms.WithThinkingMode(config.ThinkingMode),
	}

	// Track streaming output
	var streamedContent string
	streamingStartTime := time.Now()
	var firstTokenTime time.Duration
	tokenCount := 0

	if config.Streaming {
		opts = append(opts, llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			if tokenCount == 0 {
				firstTokenTime = time.Since(streamingStartTime)
			}
			tokenCount++
			streamedContent += string(chunk)
			// Print dots to show streaming progress
			fmt.Print(".")
			return nil
		}))
	}

	// For thinking modes with streaming, try to capture reasoning tokens
	if config.ThinkingMode != llms.ThinkingModeNone && config.Streaming {
		opts = append(opts, llms.WithStreamingReasoningFunc(
			func(ctx context.Context, reasoningChunk []byte, contentChunk []byte) error {
				// In real usage, you might handle reasoning chunks differently
				// For now, we just track that we received them
				if len(reasoningChunk) > 0 {
					fmt.Print("*") // Star for reasoning tokens
				}
				if len(contentChunk) > 0 {
					fmt.Print(".") // Dot for content tokens
				}
				return nil
			},
		))
	}

	fmt.Printf("Model: %s\n", model)
	fmt.Print("Generating")
	
	startTime := time.Now()
	resp, err := llm.GenerateContent(ctx, messages, opts...)
	totalTime := time.Since(startTime)

	if config.Streaming {
		fmt.Println() // New line after dots
	} else {
		fmt.Println("... done")
	}

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Display results
	content := resp.Choices[0].Content
	if len(content) > 150 {
		content = content[:150] + "..."
	}
	
	// Extract the answer if present
	answer := extractAnswer(content)
	if answer != "" {
		fmt.Printf("Answer: %s\n", answer)
	} else {
		fmt.Printf("Response: %s\n", content)
	}

	// Display metrics
	fmt.Printf("Metrics:\n")
	fmt.Printf("  Total Time: %.2fs\n", totalTime.Seconds())
	if config.Streaming && tokenCount > 0 {
		fmt.Printf("  Time to First Token: %.2fs\n", firstTokenTime.Seconds())
		fmt.Printf("  Tokens Streamed: %d\n", tokenCount)
	}

	// Display token usage
	if genInfo := resp.Choices[0].GenerationInfo; genInfo != nil {
		// Standard tokens
		if promptTokens, ok := genInfo["PromptTokens"].(int); ok {
			fmt.Printf("  Prompt Tokens: %d\n", promptTokens)
		}
		if completionTokens, ok := genInfo["CompletionTokens"].(int); ok {
			fmt.Printf("  Completion Tokens: %d\n", completionTokens)
		}
		
		// Thinking tokens (if applicable)
		usage := llms.ExtractThinkingTokens(genInfo)
		if usage != nil && usage.ThinkingTokens > 0 {
			fmt.Printf("  Thinking Tokens: %d\n", usage.ThinkingTokens)
			if usage.ThinkingBudgetAllocated > 0 {
				efficiency := float64(usage.ThinkingBudgetUsed) / float64(usage.ThinkingBudgetAllocated) * 100
				fmt.Printf("  Thinking Efficiency: %.1f%%\n", efficiency)
			}
		}
	}

	// Validate answer for the sequence problem
	expectedAnswer := "42" // 2, 6, 12, 20, 30, 42 (differences: 4, 6, 8, 10, 12)
	if strings.Contains(content, expectedAnswer) {
		fmt.Printf("  ✓ Correct answer found!\n")
	}
}

func extractAnswer(content string) string {
	// Try to extract just the number from the response
	// Look for patterns like "42", "The answer is 42", "Next number: 42", etc.
	content = strings.ToLower(content)
	
	// Common patterns
	patterns := []string{
		"answer is ", "answer: ", "next number is ", "next number: ",
		"sequence is ", "sequence: ", "therefore ", "so the answer is ",
		"the number is ", "it's ", "it is ",
	}
	
	for _, pattern := range patterns {
		if idx := strings.Index(content, pattern); idx != -1 {
			afterPattern := content[idx+len(pattern):]
			// Extract the number
			var num string
			for _, ch := range afterPattern {
				if ch >= '0' && ch <= '9' {
					num += string(ch)
				} else if len(num) > 0 {
					break
				}
			}
			if num != "" {
				return num
			}
		}
	}
	
	// If no pattern found, look for standalone "42"
	if strings.Contains(content, "42") {
		return "42"
	}
	
	return ""
}

func getEnvVarName(backendName string) string {
	switch backendName {
	case "OpenAI":
		return "OPENAI"
	case "Anthropic":
		return "ANTHROPIC"
	default:
		return strings.ToUpper(backendName)
	}
}

func checkOllamaModel(model string) bool {
	// This is a simplified check - in production you'd use Ollama's API
	// to list available models
	llm, err := ollama.New(ollama.WithModel(model))
	if err != nil {
		return false
	}
	
	// Try a minimal request to see if model exists
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	
	messages := []llms.MessageContent{{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextPart("1+1")},
	}}
	
	_, err = llm.GenerateContent(ctx, messages, llms.WithMaxTokens(1))
	return err == nil || !strings.Contains(err.Error(), "not found")
}