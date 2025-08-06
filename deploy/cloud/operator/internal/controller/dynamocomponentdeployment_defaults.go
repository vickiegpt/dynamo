package controller

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
)

// ProbeDefaults defines default probe configurations for a component type
type ProbeDefaults struct {
	Liveness  *corev1.Probe
	Readiness *corev1.Probe
}

// componentTypeDefaults holds all default configurations by component type
var componentTypeDefaults = map[string]ProbeDefaults{
	commonconsts.ComponentTypeMain: {
		Liveness: &corev1.Probe{
			InitialDelaySeconds: 20,
			PeriodSeconds:       5,
			TimeoutSeconds:      5,
			FailureThreshold:    3,
			SuccessThreshold:    1,
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: "/health",
					Port: intstr.FromInt(8000),
				},
			},
		},
		Readiness: &corev1.Probe{
			InitialDelaySeconds: 10,
			PeriodSeconds:       5,
			TimeoutSeconds:      5,
			FailureThreshold:    3,
			SuccessThreshold:    1,
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: []string{
						"/bin/sh",
						"-c",
						"curl -s http://localhost:8000/health | jq -e \".status == \\\"healthy\\\"\"",
					},
				},
			},
		},
	},
	commonconsts.ComponentTypeWorker: {
		Liveness: &corev1.Probe{
			InitialDelaySeconds: 30,
			PeriodSeconds:       10,
			TimeoutSeconds:      5,
			FailureThreshold:    3,
			SuccessThreshold:    1,
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: "/live",
					Port: intstr.FromInt(9090),
				},
			},
		},
		Readiness: &corev1.Probe{
			InitialDelaySeconds: 30,
			PeriodSeconds:       10,
			TimeoutSeconds:      5,
			FailureThreshold:    60,
			SuccessThreshold:    1,
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: "/health",
					Port: intstr.FromInt(9090),
				},
			},
		},
	},
	commonconsts.ComponentTypePlanner: {
		Liveness: &corev1.Probe{
			InitialDelaySeconds: 0,
			PeriodSeconds:       60,
			TimeoutSeconds:      30,
			FailureThreshold:    10,
			SuccessThreshold:    1,
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: []string{
						"/bin/sh",
						"-c",
						"exit 0",
					},
				},
			},
		},
		Readiness: &corev1.Probe{
			InitialDelaySeconds: 0,
			PeriodSeconds:       60,
			TimeoutSeconds:      30,
			FailureThreshold:    10,
			SuccessThreshold:    1,
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: []string{
						"/bin/sh",
						"-c",
						"exit 0",
					},
				},
			},
		},
	},
}

// fallbackDefaults are used when component type is unknown
var fallbackDefaults = ProbeDefaults{
	Liveness: &corev1.Probe{
		InitialDelaySeconds: 60,
		PeriodSeconds:       60,
		TimeoutSeconds:      5,
		FailureThreshold:    10,
		SuccessThreshold:    1,
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/healthz",
				Port: intstr.FromString(commonconsts.DynamoHealthPortName),
			},
		},
	},
	Readiness: &corev1.Probe{
		InitialDelaySeconds: 60,
		PeriodSeconds:       60,
		TimeoutSeconds:      5,
		FailureThreshold:    10,
		SuccessThreshold:    1,
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/readyz",
				Port: intstr.FromString(commonconsts.DynamoHealthPortName),
			},
		},
	},
}

// getEffectiveLivenessProbe returns the effective liveness probe after merging defaults with user overrides
func (r *DynamoComponentDeploymentReconciler) getEffectiveLivenessProbe(deployment *v1alpha1.DynamoComponentDeployment) *corev1.Probe {
	defaults, exists := componentTypeDefaults[deployment.Spec.ComponentType]
	if !exists {
		defaults = fallbackDefaults
	}
	return mergeProbe(defaults.Liveness, deployment.Spec.LivenessProbe)
}

// getEffectiveReadinessProbe returns the effective readiness probe after merging defaults with user overrides
func (r *DynamoComponentDeploymentReconciler) getEffectiveReadinessProbe(deployment *v1alpha1.DynamoComponentDeployment) *corev1.Probe {
	defaults, exists := componentTypeDefaults[deployment.Spec.ComponentType]
	if !exists {
		defaults = fallbackDefaults
	}
	return mergeProbe(defaults.Readiness, deployment.Spec.ReadinessProbe)
}

// mergeProbe merges user-provided probe overrides with default probe configuration
func mergeProbe(defaultProbe, userProbe *corev1.Probe) *corev1.Probe {
	if userProbe == nil {
		return defaultProbe.DeepCopy()
	}

	// Start with a deep copy of the default probe
	result := defaultProbe.DeepCopy()

	// Override timing fields if explicitly set by user
	if userProbe.InitialDelaySeconds != 0 {
		result.InitialDelaySeconds = userProbe.InitialDelaySeconds
	}
	if userProbe.PeriodSeconds != 0 {
		result.PeriodSeconds = userProbe.PeriodSeconds
	}
	if userProbe.TimeoutSeconds != 0 {
		result.TimeoutSeconds = userProbe.TimeoutSeconds
	}
	if userProbe.FailureThreshold != 0 {
		result.FailureThreshold = userProbe.FailureThreshold
	}
	if userProbe.SuccessThreshold != 0 {
		result.SuccessThreshold = userProbe.SuccessThreshold
	}

	// Handle probe handler overrides - they are mutually exclusive
	switch {
	case userProbe.HTTPGet != nil:
		result.ProbeHandler = mergeHTTPGetHandler(result.HTTPGet, userProbe.HTTPGet)
	case userProbe.Exec != nil:
		result.ProbeHandler = corev1.ProbeHandler{Exec: userProbe.Exec}
	case userProbe.TCPSocket != nil:
		result.ProbeHandler = mergeTCPSocketHandler(result.TCPSocket, userProbe.TCPSocket)
	case userProbe.GRPC != nil:
		result.ProbeHandler = corev1.ProbeHandler{GRPC: userProbe.GRPC}
	}

	return result
}

// mergeHTTPGetHandler merges HTTPGet probe handlers, allowing partial field overrides
func mergeHTTPGetHandler(defaultHTTP, userHTTP *corev1.HTTPGetAction) corev1.ProbeHandler {
	if defaultHTTP == nil {
		return corev1.ProbeHandler{HTTPGet: userHTTP}
	}

	merged := defaultHTTP.DeepCopy()
	if userHTTP.Path != "" {
		merged.Path = userHTTP.Path
	}
	if userHTTP.Port.IntVal != 0 || userHTTP.Port.StrVal != "" {
		merged.Port = userHTTP.Port
	}
	if userHTTP.Host != "" {
		merged.Host = userHTTP.Host
	}
	if userHTTP.Scheme != "" {
		merged.Scheme = userHTTP.Scheme
	}
	if userHTTP.HTTPHeaders != nil {
		merged.HTTPHeaders = userHTTP.HTTPHeaders
	}

	return corev1.ProbeHandler{HTTPGet: merged}
}

// mergeTCPSocketHandler merges TCPSocket probe handlers, allowing partial field overrides
func mergeTCPSocketHandler(defaultTCP, userTCP *corev1.TCPSocketAction) corev1.ProbeHandler {
	if defaultTCP == nil {
		return corev1.ProbeHandler{TCPSocket: userTCP}
	}

	merged := defaultTCP.DeepCopy()
	if userTCP.Port.IntVal != 0 || userTCP.Port.StrVal != "" {
		merged.Port = userTCP.Port
	}
	if userTCP.Host != "" {
		merged.Host = userTCP.Host
	}

	return corev1.ProbeHandler{TCPSocket: merged}
}
