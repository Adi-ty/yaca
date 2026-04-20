package ai

import (
	"fmt"
	"sync"
)

var registry struct {
	mu        sync.RWMutex
	providers map[string]Provider
}

// Register adds a provider to the global registry.
// It panics if a provider with the same name has already been registered,
// matching the convention used by database/sql drivers.
func Register(p Provider) {
	registry.mu.Lock()
	defer registry.mu.Unlock()

	if registry.providers == nil {
		registry.providers = make(map[string]Provider)
	}

	name := p.Name()
	if _, exists := registry.providers[name]; exists {
		panic(fmt.Sprintf("ai: provider %q already registered", name))
	}
	registry.providers[name] = p
}

// Get returns the provider registered under name, or an error if not found.
func Get(name string) (Provider, error) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	p, ok := registry.providers[name]
	if !ok {
		return nil, fmt.Errorf("ai: provider %q not registered", name)
	}
	return p, nil
}

// All returns a snapshot of all registered providers, keyed by name.
func All() map[string]Provider {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	out := make(map[string]Provider, len(registry.providers))
	for k, v := range registry.providers {
		out[k] = v
	}
	return out
}
