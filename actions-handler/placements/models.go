package placements

import "fmt"

type Workload struct {
	APIVersion string `json:"apiVersion" binding:"required" validate:"required"`
	Kind       string `json:"kind" binding:"required" validate:"required"`
	Namespace  string `json:"namespace" binding:"required" validate:"required"`
	Name       string `json:"name" binding:"required" validate:"required"`
}

func (w Workload) GetId() string {
	return fmt.Sprintf("%s-%s-%s-%s",
		w.APIVersion,
		w.Kind,
		w.Namespace,
		w.Name,
	)
}

type Placement struct {
	Timestamp   *int64   `json:"timestamp,omitempty" binding:"omitempty"`
	Workload    Workload `json:"workload" binding:"required" validate:"required,dive"`
	ClusterName string   `json:"clusterName" binding:"required" validate:"required"`
	Replicas    int32    `json:"replicas" binding:"omitempty" validate:"omitempty"`
}
