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

package controller

import (
	"context"
	"fmt"
	karmadapolicyv1alpha1 "github.com/karmada-io/api/policy/v1alpha1"
	"github.com/pkg/errors"
	appsv1 "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"reflect"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sort"
	"time"

	policyv1alpha1 "gitlab.eclipse.org/eclipse-research-labs/nephele-project/opencall-2/headlight/controlled-replica-propagation/api/v1alpha1"
)

// ControlledReplicaPropagationReconciler reconciles a ControlledReplicaPropagation object
type ControlledReplicaPropagationReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=policy.nephele-project.eu,resources=controlledreplicapropagations,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=policy.karmada.io,resources=propagationpolicies,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// Modify the Reconcile function to compare the state specified by
// the ControlledReplicaPropagation object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.20.4/pkg/reconcile
func (r *ControlledReplicaPropagationReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	crp := &policyv1alpha1.ControlledReplicaPropagation{}
	if err := r.Get(ctx, req.NamespacedName, crp); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info(
				fmt.Sprintf(
					"ControlledReplicaPropagation %s in namespace %s not found. Ignoring since it must have been deleted.",
					req.NamespacedName.Name,
					req.NamespacedName.Namespace,
				),
			)
			return ctrl.Result{}, nil
		}

		log.Error(err, fmt.Sprintf(
			"Failed to get ControlledReplicaPropagation %s in namespace %s.",
			req.NamespacedName.Name,
			req.NamespacedName.Namespace,
		),
		)
		return ctrl.Result{}, err
	}

	strategy := crp.Spec.Strategy

	// If no strategy is specified,
	// default to a non-live migration strategy.
	// todo: this should probably be part of mutating webhook.
	if strategy == nil {
		totalCrpReplicas := crp.GetTotalReplicas()
		strategy = &policyv1alpha1.Strategy{
			MaxUnavailable: &totalCrpReplicas,
		}
	}

	foundDep := &appsv1.Deployment{}
	targetDepNamespace := crp.Namespace
	targetDepName := crp.Spec.TargetRef.Name
	err := r.Get(ctx, types.NamespacedName{Namespace: targetDepNamespace, Name: targetDepName}, foundDep)
	if err != nil && apierrors.IsNotFound(err) {
		log.Error(err, fmt.Sprintf("Target Deployment %s in namespace %s not found.", targetDepName, targetDepNamespace))
		return ctrl.Result{}, err
	}
	if err != nil {
		log.Error(err, fmt.Sprintf("Error getting targeted Deployment %s in namespace %s.", targetDepName, targetDepNamespace))
		return ctrl.Result{}, err
	}

	// If deployment is upgrading, wait for it to finish
	// before making any placement/scaling changes.
	if !reflect.ValueOf(foundDep.Status).IsZero() &&
		foundDep.Status.UpdatedReplicas < foundDep.Status.Replicas {
		log.Info("Deployment is upgrading, waiting until upgrade completes.", "Deployment.Namespace", foundDep.Namespace, "Deployment.Name", foundDep.Name)
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}

	foundPolicy := &karmadapolicyv1alpha1.PropagationPolicy{}
	err = r.Get(ctx, types.NamespacedName{Name: fmt.Sprintf("%s-%s", "crp-owned", crp.Name), Namespace: crp.Namespace}, foundPolicy)
	if err != nil && apierrors.IsNotFound(err) {
		// Create a missing PropagationPolicy.
		policy, err := r.newPropagationPolicy(crp)
		if err != nil {
			log.Error(err, "Failed to create new PropagationPolicy", "PropagationPolicy.Namespace", policy.Namespace, "PropagationPolicy.Name", policy.Name)
			return ctrl.Result{}, err
		}

		log.Info("Creating a new PropagationPolicy", "PropagationPolicy.Namespace", policy.Namespace, "PropagationPolicy.Name", policy.Name)
		if err := r.Create(ctx, policy); err != nil {
			log.Error(err, "Failed to create new PropagationPolicy", "PropagationPolicy.Namespace", policy.Namespace, "PropagationPolicy.Name", policy.Name)
			return ctrl.Result{}, err
		}
		// Requeue the request to ensure the PropagationPolicy is created.
		return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
	} else if err != nil {
		log.Error(err, "Failed to get PropagationPolicy")
		return ctrl.Result{}, err
	}

	// todo: validating webhook has to ensure only one of them is not nil
	switch {
	case strategy.MaxUnavailable != nil:
		return r.reconcileMaxUnavailable(ctx, *strategy.MaxUnavailable, crp, foundDep, foundPolicy)
	default:
		return ctrl.Result{}, errors.New("unsupported strategy provided")
	}
}

func (r *ControlledReplicaPropagationReconciler) reconcileMaxUnavailable(
	ctx context.Context,
	maxUnavailable int32,
	crp *policyv1alpha1.ControlledReplicaPropagation,
	dep *appsv1.Deployment,
	policy *karmadapolicyv1alpha1.PropagationPolicy,
) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	// We initially attempted an optimal strategy where replicas were deleted
	// as soon as there was one more available than the total available desired.
	// However, this led to issues, likely due to the propagation policy
	// not being designed for rapid, incremental updates
	// (weights not replicas, non-atomic updates to policies and deployments).
	//
	// Specifically, we observed replicas being prematurely removed before live-migration
	// could complete, breaking availability guarantees.
	//
	// To address this, we implement a different approach:
	// 1. Create all newly needed replicas, ignoring deletes for now.
	// 2. Once the new replicas are available (minus maxUnavailable), proceed with deletions.
	//
	// This ensures that the number of available replicas never drops below the desired count.
	// The trade-off is that we must wait for all new replicas to become available before
	// performing any deletes.

	desiredPlacements := crp.Spec.Placements
	currentPlacements, err := karmadaPlacementsToPlacements(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList)
	if err != nil {
		log.Error(err, "Failed to convert karmada placements to placements")
		return ctrl.Result{}, err
	}

	desiredPlacementsMap := placementsToMap(desiredPlacements)
	currentPlacementsMap := placementsToMap(currentPlacements)

	updatedDepReplicas := int32(0)
	updatedPlacements := make([]policyv1alpha1.Placement, 0, len(desiredPlacements))

	hasCreates := false
	for clusterName, desiredClusterReplicas := range desiredPlacementsMap {
		// If in desired, but not in current, then add replicas.
		if _, ok := currentPlacementsMap[clusterName]; !ok {
			updatedPlacements = append(updatedPlacements, policyv1alpha1.Placement{
				ClusterName: clusterName,
				Replicas:    desiredClusterReplicas,
			})
			updatedDepReplicas += desiredClusterReplicas
			hasCreates = true
			continue
		}

		// If in desired and in current and desired replicas > current replicas,
		// then add replicas.
		currentClusterReplicas := currentPlacementsMap[clusterName]
		if desiredClusterReplicas > currentClusterReplicas {
			updatedPlacements = append(updatedPlacements, policyv1alpha1.Placement{
				ClusterName: clusterName,
				Replicas:    desiredClusterReplicas,
			})
			updatedDepReplicas += desiredClusterReplicas
			hasCreates = true
		} else {
			// Realize only creates first,
			// so keep other placements as is for now.
			updatedPlacements = append(updatedPlacements, policyv1alpha1.Placement{
				ClusterName: clusterName,
				Replicas:    currentClusterReplicas,
			})
			updatedDepReplicas += currentClusterReplicas
		}
	}

	// Ignore removes for now.
	for clusterName, currentClusterReplicas := range currentPlacementsMap {
		if _, ok := desiredPlacementsMap[clusterName]; !ok {
			updatedPlacements = append(updatedPlacements, policyv1alpha1.Placement{
				ClusterName: clusterName,
				Replicas:    currentClusterReplicas,
			})
			updatedDepReplicas += currentClusterReplicas
		}
	}

	// If there are no creates and all replicas are available
	// that means it's safe to remove replicas as desired.
	if !hasCreates && dep.Status.AvailableReplicas+maxUnavailable >= *dep.Spec.Replicas {
		updatedDepReplicas = crp.GetTotalReplicas()
		updatedPlacements = desiredPlacements
	}

	currentDepReplicas := *dep.Spec.Replicas
	hasReplicasChanged := currentDepReplicas != updatedDepReplicas

	sortPlacements(currentPlacements)
	sortPlacements(updatedPlacements)
	hasPlacementsChanged := !arePlacementsEqual(currentPlacements, updatedPlacements)

	isPolicySuspended := false
	if policy.Spec.Suspension != nil && policy.Spec.Suspension.Dispatching != nil {
		isPolicySuspended = *policy.Spec.Suspension.Dispatching
	}

	// If there are changes, suspend propagation policy.
	// It seems to help reduce unnecessary temporary replicas creation
	// while non-atomic patches are applied.
	if (hasReplicasChanged || hasPlacementsChanged) && !isPolicySuspended {
		original := client.MergeFrom(policy.DeepCopy())
		isSuspended := true
		policy.Spec.Suspension = &karmadapolicyv1alpha1.Suspension{
			Dispatching: &isSuspended,
		}
		log.Info("Suspending PropagationPolicy", "PropagationPolicy.Namespace", policy.Namespace, "PropagationPolicy.Name", policy.Name)
		err = r.Patch(ctx, policy, original)
		if err != nil {
			log.Error(err, "Failed to suspend PropagationPolicy", "PropagationPolicy.Namespace", policy.Namespace, "PropagationPolicy.Name", policy.Name)
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Deployment and propagation policy updates are not atomic.
	// Because the propagation policy relies on weights, it's possible that,
	// momentarily, more replicas are created or removed than intended during updates.
	// While temporary over-provisioning is acceptable and quickly corrected,
	// under-provisioning is not.
	//
	// To avoid this, we order the updates to ensure that, in the worst case,
	// we briefly have extra replicas, not fewer than required.
	// Temporarily suspending the propagation policy also seems to help with this.
	if updatedDepReplicas < currentDepReplicas {
		if hasPlacementsChanged {
			log.Info("Patching PropagationPolicy with new placements", "Placement.Replicas", updatedDepReplicas, "Placement.ClusterNames", updatedPlacements)
			err := r.patchPlacements(ctx, policy, updatedPlacements)
			if err != nil {
				log.Error(err, "Failed to patch PropagationPolicy.", "PropagationPolicy.Namespace", policy.Namespace, "PropagationPolicy.Name", policy.Name)
				return ctrl.Result{}, err
			}
		}

		if hasReplicasChanged {
			log.Info("Patching Deployment with new replicas", "Deployment.Replicas", updatedDepReplicas)
			err := r.patchReplicas(ctx, dep, updatedDepReplicas)
			if err != nil {
				log.Error(err, "Failed to patch Deployment replicas.", "Deployment.Namespace", dep.Namespace, "Deployment.Name", dep.Name)
				return ctrl.Result{}, err
			}
		}
	} else {
		if hasReplicasChanged {
			log.Info("Patching Deployment with new replicas", "Deployment.Replicas", updatedDepReplicas)
			err := r.patchReplicas(ctx, dep, updatedDepReplicas)
			if err != nil {
				log.Error(err, "Failed to patch Deployment replicas.", "Deployment.Namespace", dep.Namespace, "Deployment.Name", dep.Name)
				return ctrl.Result{}, err
			}
		}

		if hasPlacementsChanged {
			log.Info("Patching PropagationPolicy with new placements", "Placement.Replicas", updatedDepReplicas, "Placement.ClusterNames", updatedPlacements)
			err := r.patchPlacements(ctx, policy, updatedPlacements)
			if err != nil {
				log.Error(err, "Failed to patch PropagationPolicy.", "PropagationPolicy.Namespace", policy.Namespace, "PropagationPolicy.Name", policy.Name)
				return ctrl.Result{}, err
			}
		}
	}

	if hasPlacementsChanged || hasReplicasChanged {
		return ctrl.Result{Requeue: true}, nil
	}

	// If we get here, then there are no changes to make.
	// Make sure the propagation policy is unsuspended.
	if isPolicySuspended {
		original := client.MergeFrom(policy.DeepCopy())
		isSuspended := false
		policy.Spec.Suspension = &karmadapolicyv1alpha1.Suspension{
			Dispatching: &isSuspended,
		}

		log.Info("Unsuspending PropagationPolicy", "PropagationPolicy.Namespace", policy.Namespace, "PropagationPolicy.Name", policy.Name)
		err = r.Patch(ctx, policy, original)
		if err != nil {
			log.Error(err, "Failed to unsuspend PropagationPolicy", "PropagationPolicy.Namespace", policy.Namespace, "PropagationPolicy.Name", policy.Name)
			return ctrl.Result{RequeueAfter: 10 * time.Second}, err
		}
	}

	return ctrl.Result{}, nil
}

func (r *ControlledReplicaPropagationReconciler) patchPlacements(ctx context.Context, policy *karmadapolicyv1alpha1.PropagationPolicy, placements []policyv1alpha1.Placement) error {
	original := client.MergeFrom(policy.DeepCopy())
	karmadaPlacements := placementsToKarmadaPlacements(placements)
	policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList = karmadaPlacements

	err := r.Patch(ctx, policy, original)
	if err != nil {
		return errors.Wrap(err, "failed to patch placements")
	}

	return nil
}

func (r *ControlledReplicaPropagationReconciler) patchReplicas(ctx context.Context, dep *appsv1.Deployment, replicas int32) error {
	original := client.MergeFrom(dep.DeepCopy())
	dep.Spec.Replicas = &replicas

	err := r.Patch(ctx, dep, original)
	if err != nil {
		return errors.Wrap(err, "failed to patch replicas")
	}

	return nil
}

func (r *ControlledReplicaPropagationReconciler) newPropagationPolicy(crp *policyv1alpha1.ControlledReplicaPropagation) (*karmadapolicyv1alpha1.PropagationPolicy, error) {
	policy := &karmadapolicyv1alpha1.PropagationPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%s", "crp-owned", crp.Name),
			Namespace: crp.Namespace,
		},
		Spec: karmadapolicyv1alpha1.PropagationSpec{
			ResourceSelectors: []karmadapolicyv1alpha1.ResourceSelector{
				{
					APIVersion: crp.Spec.TargetRef.APIVersion,
					Kind:       crp.Spec.TargetRef.Kind,
					Namespace:  crp.Namespace,
					Name:       crp.Spec.TargetRef.Name,
				},
			},
			Placement: karmadapolicyv1alpha1.Placement{
				ReplicaScheduling: &karmadapolicyv1alpha1.ReplicaSchedulingStrategy{
					ReplicaSchedulingType:     "Divided",
					ReplicaDivisionPreference: "Weighted",
					WeightPreference:          &karmadapolicyv1alpha1.ClusterPreferences{},
				},
			},
		},
	}

	placements := crp.Spec.Placements
	sortPlacements(placements)

	karmadaPlacement := make([]karmadapolicyv1alpha1.StaticClusterWeight, 0, len(placements))
	for _, placement := range placements {
		karmadaPlacement = append(karmadaPlacement, karmadapolicyv1alpha1.StaticClusterWeight{
			TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
				ClusterNames: []string{
					placement.ClusterName,
				},
			},
			Weight: int64(placement.Replicas),
		})
	}

	policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList = karmadaPlacement

	err := controllerutil.SetControllerReference(crp, policy, r.Scheme)
	if err != nil {
		return nil, errors.Wrap(err, "failed to set controller reference")
	}

	return policy, nil
}

func placementsToMap(placements []policyv1alpha1.Placement) map[string]int32 {
	placementsMap := make(map[string]int32)
	for _, placement := range placements {
		placementsMap[placement.ClusterName] = placement.Replicas
	}

	return placementsMap
}

func sortPlacements(placements []policyv1alpha1.Placement) {
	sort.Slice(placements, func(i, j int) bool {
		return placements[i].ClusterName < placements[j].ClusterName
	})
}

func arePlacementsEqual(a, b []policyv1alpha1.Placement) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i].ClusterName != b[i].ClusterName ||
			a[i].Replicas != b[i].Replicas {
			return false
		}
	}

	return true
}

func karmadaPlacementsToPlacements(karmadaPlacements []karmadapolicyv1alpha1.StaticClusterWeight) ([]policyv1alpha1.Placement, error) {
	placements := make([]policyv1alpha1.Placement, 0, len(karmadaPlacements))

	for _, karmadaPlacement := range karmadaPlacements {

		if karmadaPlacement.TargetCluster.ClusterNames == nil ||
			len(karmadaPlacement.TargetCluster.ClusterNames) == 0 ||
			len(karmadaPlacement.TargetCluster.ClusterNames) > 1 {
			return nil, errors.New(fmt.Sprintf("expected exactly one cluster name in karmada static weight, got: %v", karmadaPlacement.TargetCluster.ClusterNames))
		}

		if karmadaPlacement.Weight <= 0 {
			return nil, errors.New(fmt.Sprintf("expected weight to be greater than 0 in karmada static weight, got: %v", karmadaPlacement.Weight))
		}

		placements = append(placements, policyv1alpha1.Placement{
			ClusterName: karmadaPlacement.TargetCluster.ClusterNames[0],
			Replicas:    int32(karmadaPlacement.Weight),
		})
	}

	return placements, nil
}

func placementsToKarmadaPlacements(placements []policyv1alpha1.Placement) []karmadapolicyv1alpha1.StaticClusterWeight {
	karmadaPlacements := make([]karmadapolicyv1alpha1.StaticClusterWeight, 0, len(placements))
	for _, placement := range placements {
		karmadaPlacements = append(karmadaPlacements, karmadapolicyv1alpha1.StaticClusterWeight{
			TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
				ClusterNames: []string{
					placement.ClusterName,
				},
			},
			Weight: int64(placement.Replicas),
		})
	}

	return karmadaPlacements
}

// SetupWithManager sets up the controller with the Manager.
func (r *ControlledReplicaPropagationReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&policyv1alpha1.ControlledReplicaPropagation{}).
		Named("controlledreplicapropagation").
		Owns(&karmadapolicyv1alpha1.PropagationPolicy{}).
		Watches(
			&appsv1.Deployment{},
			handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []reconcile.Request {
				log := mgr.GetLogger()

				crpList := &policyv1alpha1.ControlledReplicaPropagationList{}
				if err := mgr.GetClient().List(ctx, crpList); err != nil {
					log.Error(err, "Error listing ControlledReplicaPropagations")
					return nil
				}

				reqs := make([]ctrl.Request, 0, len(crpList.Items))
				for _, crp := range crpList.Items {
					if crp.GetNamespace() == obj.GetNamespace() &&
						crp.Spec.TargetRef.Name == obj.GetName() {
						reqs = append(reqs, ctrl.Request{
							NamespacedName: types.NamespacedName{
								Namespace: crp.GetNamespace(),
								Name:      crp.GetName(),
							},
						})
					}
				}

				return reqs
			}),
		).
		Complete(r)
}
