/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestCalculateDynamoNamespace(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name         string
		dgd          *v1alpha1.DynamoGraphDeployment
		expected     string
		expectError  bool
		errorMessage string
	}{
		{
			name: "default namespace when no services",
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{},
				},
			},
			expected:    "dynamo-test-dgd",
			expectError: false,
		},
		{
			name: "default namespace when no custom namespace specified",
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
						"frontend": {
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								DynamoNamespace: nil,
							},
						},
						"backend": {
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								DynamoNamespace: nil,
							},
						},
					},
				},
			},
			expected:    "dynamo-test-dgd",
			expectError: false,
		},
		{
			name: "custom namespace when specified",
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
						"frontend": {
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								DynamoNamespace: ptr.To("inference"),
							},
						},
						"backend": {
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								DynamoNamespace: ptr.To("inference"),
							},
						},
					},
				},
			},
			expected:    "inference",
			expectError: false,
		},
		{
			name: "error when namespace mismatch",
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
						"frontend": {
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								DynamoNamespace: ptr.To("inference"),
							},
						},
						"backend": {
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								DynamoNamespace: ptr.To("training"),
							},
						},
					},
				},
			},
			expected:     "",
			expectError:  true,
			errorMessage: "namespace mismatch for component backend: graph uses namespace inference but component specifies training",
		},
		{
			name: "empty string namespace falls back to default",
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
						"frontend": {
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								DynamoNamespace: ptr.To(""),
							},
						},
					},
				},
			},
			expected:    "dynamo-test-dgd",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := calculateDynamoNamespace(ctx, tt.dgd)

			if tt.expectError {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMessage)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

func TestCalculateComponentLabels(t *testing.T) {
	tests := []struct {
		name     string
		services map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec
		expected map[string]string
	}{
		{
			name:     "empty services",
			services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			expected: map[string]string{},
		},
		{
			name: "single service",
			services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
				"frontend": {},
			},
			expected: map[string]string{
				"nvidia.com/manages-component-frontend": "true",
			},
		},
		{
			name: "multiple services",
			services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
				"frontend":   {},
				"vllmWorker": {},
				"planner":    {},
			},
			expected: map[string]string{
				"nvidia.com/manages-component-frontend":   "true",
				"nvidia.com/manages-component-vllmWorker": "true",
				"nvidia.com/manages-component-planner":    "true",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateComponentLabels(tt.services)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestExtractExistingComponentLabels(t *testing.T) {
	tests := []struct {
		name     string
		labels   map[string]string
		expected map[string]string
	}{
		{
			name:     "nil labels",
			labels:   nil,
			expected: map[string]string{},
		},
		{
			name:     "empty labels",
			labels:   map[string]string{},
			expected: map[string]string{},
		},
		{
			name: "no component labels",
			labels: map[string]string{
				"app":                         "my-app",
				"nvidia.com/dynamo-namespace": "inference",
				"kubernetes.io/managed-by":    "dynamo",
			},
			expected: map[string]string{},
		},
		{
			name: "mixed labels with component labels",
			labels: map[string]string{
				"app":                                     "my-app",
				"nvidia.com/dynamo-namespace":             "inference",
				"nvidia.com/manages-component-frontend":   "true",
				"nvidia.com/manages-component-vllmWorker": "true",
				"kubernetes.io/managed-by":                "dynamo",
			},
			expected: map[string]string{
				"nvidia.com/manages-component-frontend":   "true",
				"nvidia.com/manages-component-vllmWorker": "true",
			},
		},
		{
			name: "only component labels",
			labels: map[string]string{
				"nvidia.com/manages-component-frontend": "true",
				"nvidia.com/manages-component-backend":  "true",
				"nvidia.com/manages-component-planner":  "true",
			},
			expected: map[string]string{
				"nvidia.com/manages-component-frontend": "true",
				"nvidia.com/manages-component-backend":  "true",
				"nvidia.com/manages-component-planner":  "true",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractExistingComponentLabels(tt.labels)
			assert.Equal(t, tt.expected, result)
		})
	}
}
