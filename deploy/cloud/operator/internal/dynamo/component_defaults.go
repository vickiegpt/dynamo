/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// ComponentDefaults interface defines how defaults should be provided
type ComponentDefaults interface {
	// GetBaseContainer returns the base container configuration for this component type
	// The backendFramework parameter may be empty for components that don't need backend-specific config
	GetBaseContainer(backendFramework BackendFramework) (corev1.Container, error)
}

// ComponentDefaultsFactory creates appropriate defaults based on component type
func ComponentDefaultsFactory(componentType string) ComponentDefaults {
	switch componentType {
	case commonconsts.ComponentTypeMain:
		return NewFrontendDefaults()
	case commonconsts.ComponentTypeWorker:
		return NewWorkerDefaults()
	default:
		return &BaseComponentDefaults{}
	}
}
