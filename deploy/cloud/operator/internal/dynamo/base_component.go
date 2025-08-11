/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	corev1 "k8s.io/api/core/v1"
)

// BaseComponentDefaults provides common defaults shared by all components
type BaseComponentDefaults struct{}

func (b *BaseComponentDefaults) GetBaseContainer(backendFramework BackendFramework) (corev1.Container, error) {
	return b.getCommonContainer(), nil
}

func (b *BaseComponentDefaults) getCommonContainer() corev1.Container {
	container := corev1.Container{
		Name: "main",
	}

	return container
}
