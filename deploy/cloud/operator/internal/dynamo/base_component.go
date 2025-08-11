/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// BaseComponentDefaults provides common defaults shared by all components
type BaseComponentDefaults struct{}

func (b *BaseComponentDefaults) GetBaseContainer(backendFramework BackendFramework) (corev1.Container, error) {
	return b.getCommonContainer(), nil
}

func (b *BaseComponentDefaults) getCommonContainer() corev1.Container {
	return corev1.Container{
		Name: "main",
		Ports: []corev1.ContainerPort{
			{
				Protocol:      corev1.ProtocolTCP,
				Name:          commonconsts.DynamoContainerPortName,
				ContainerPort: int32(commonconsts.DynamoServicePort),
			},
			{
				Protocol:      corev1.ProtocolTCP,
				Name:          commonconsts.DynamoHealthPortName,
				ContainerPort: int32(commonconsts.DynamoHealthPort),
			},
		},
	}
}
