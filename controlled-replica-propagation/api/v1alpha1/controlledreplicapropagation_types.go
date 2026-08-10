/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// ControlledReplicaPropagationSpec defines the desired state of ControlledReplicaPropagation.
type ControlledReplicaPropagationSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// +optional
	Strategy *Strategy `json:"strategy,omitempty"`

	TargetRef CrossVersionObjectReference `json:"targetRef"`

	// todo: ensure this is a set (maybe in validation webhook)
	Placements []Placement `json:"placements"`
}

// Strategy defines the strategy for propagating replicas across clusters in the federation.
// If no strategy is specified, the default strategy maxUnavailable=replicas will be used.
type Strategy struct {
	// This ensures >= (replicas - maxUnavailable) available replicas
	// on the federation level at any point in time
	// +optional
	// +kubebuilder:validation:Minimum=0
	MaxUnavailable *int32 `json:"maxUnavailable,omitempty"`
}

// Placement defines replica distribution across clusters
type Placement struct {
	ClusterName string `json:"clusterName"`
	// +kubebuilder:validation:Minimum=1
	Replicas int32 `json:"replicas"`
}

// CrossVersionObjectReference contains enough information to let you identify the referred resource.
type CrossVersionObjectReference struct {
	// kind is the kind of the referent; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	Kind string `json:"kind"`

	// name is the name of the referent; More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#names
	Name string `json:"name"`

	// apiVersion is the API version of the referent
	// +optional
	APIVersion string `json:"apiVersion,omitempty"`
}

// ControlledReplicaPropagationStatus defines the observed state of ControlledReplicaPropagation.
type ControlledReplicaPropagationStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
}

func (c *ControlledReplicaPropagation) GetTotalReplicas() int32 {
	total := int32(0)
	for _, placement := range c.Spec.Placements {
		total += placement.Replicas
	}

	return total
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// ControlledReplicaPropagation is the Schema for the controlledreplicapropagations API.
type ControlledReplicaPropagation struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ControlledReplicaPropagationSpec   `json:"spec,omitempty"`
	Status ControlledReplicaPropagationStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ControlledReplicaPropagationList contains a list of ControlledReplicaPropagation.
type ControlledReplicaPropagationList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ControlledReplicaPropagation `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ControlledReplicaPropagation{}, &ControlledReplicaPropagationList{})
}
