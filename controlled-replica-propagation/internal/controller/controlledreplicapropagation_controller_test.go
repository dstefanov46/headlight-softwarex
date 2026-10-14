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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	policyv1alpha1 "gitlab.eclipse.org/eclipse-research-labs/nephele-project/opencall-2/headlight/controlled-replica-propagation/api/v1alpha1"
)

var _ = Describe("ControlledReplicaPropagation Controller", func() {
	When("creating a new ControlledReplicaPropagation with existing target Deployment", func() {
		const resourceName = "test-create-resource"
		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}
		typeNamespacedNamePolicy := types.NamespacedName{
			Name:      fmt.Sprintf("%s-%s", "crp-owned", resourceName),
			Namespace: "default",
		}

		BeforeEach(func() {
			dep := &appsv1.Deployment{}
			err := k8sClient.Get(ctx, typeNamespacedName, dep)
			if err != nil && errors.IsNotFound(err) {
				replicas := int32(1)
				resource := &appsv1.Deployment{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &replicas,
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"app": "test"},
						},
						Template: v1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{"app": "test"},
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Name:  "test",
										Image: "alpine",
									},
								},
							},
						},
					},
				}

				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}

			Expect(k8sClient.Get(ctx, typeNamespacedName, dep)).To(Succeed())

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err = k8sClient.Get(ctx, typeNamespacedNamePolicy, crp)
			if err != nil && errors.IsNotFound(err) {
				maxUnavailable := int32(0)
				resource := &policyv1alpha1.ControlledReplicaPropagation{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
					},
					Spec: policyv1alpha1.ControlledReplicaPropagationSpec{
						Strategy: &policyv1alpha1.Strategy{
							MaxUnavailable: &maxUnavailable,
						},
						TargetRef: policyv1alpha1.CrossVersionObjectReference{
							APIVersion: dep.APIVersion,
							Kind:       dep.Kind,
							Name:       dep.Name,
						},
						Placements: []policyv1alpha1.Placement{
							{
								ClusterName: "cluster1",
								Replicas:    5,
							},
							{
								ClusterName: "cluster2",
								Replicas:    5,
							},
							{
								ClusterName: "cluster3",
								Replicas:    5,
							},
						},
					},
				}

				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}
		})

		AfterEach(func() {
			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err := k8sClient.Get(ctx, typeNamespacedName, crp)
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Delete(ctx, crp)).To(Succeed())

			dep := &appsv1.Deployment{}
			err = k8sClient.Get(ctx, typeNamespacedName, dep)
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Delete(ctx, dep)).To(Succeed())
		})

		It("should successfully create the Karmada PropagationPolicy", func() {
			mustReconcileTimes(1, typeNamespacedName)

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, typeNamespacedNamePolicy, policy)).To(Succeed())

			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 5,
				},
			}))
		})

		It("should suspend the Karmada PropagationPolicy", func() {
			mustReconcileTimes(2, typeNamespacedName)

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, typeNamespacedNamePolicy, policy)).To(Succeed())

			Expect(*policy.Spec.Suspension.Dispatching).To(BeTrue())
		})

		It("should update the Deployment's replicas", func() {
			mustReconcileTimes(3, typeNamespacedName)

			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, dep)).To(Succeed())

			Expect(*dep.Spec.Replicas).To(Equal(int32(15)))
		})

		It("should unsuspend the Karmada PropagationPolicy", func() {
			mustReconcileTimes(3, typeNamespacedName)

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, typeNamespacedNamePolicy, policy)).To(Succeed())

			Expect(*policy.Spec.Suspension.Dispatching).To(BeFalse())
		})
	})

	When("creating a new ControlledReplicaPropagation with a non-existent target Deployment", func() {
		const resourceName = "test-no-dep-resource"
		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}

		BeforeEach(func() {
			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err := k8sClient.Get(ctx, typeNamespacedName, crp)
			if err != nil && errors.IsNotFound(err) {
				maxUnavailable := int32(0)
				resource := &policyv1alpha1.ControlledReplicaPropagation{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
					},
					Spec: policyv1alpha1.ControlledReplicaPropagationSpec{
						Strategy: &policyv1alpha1.Strategy{
							MaxUnavailable: &maxUnavailable,
						},
						TargetRef: policyv1alpha1.CrossVersionObjectReference{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "non-existent-deployment",
						},
						Placements: []policyv1alpha1.Placement{},
					},
				}

				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}
		})

		AfterEach(func() {
			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err := k8sClient.Get(ctx, typeNamespacedName, crp)
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Delete(ctx, crp)).To(Succeed())
		})

		It("should error", func() {
			err := failsOnReconcile(1, typeNamespacedName)
			Expect(err.Error()).To(ContainSubstring("not found"))
		})
	})

	When("when scaling an individual cluster", func() {
		ctx := context.Background()

		setupInitState := func(uniqueName string) types.NamespacedName {
			namespacedName := types.NamespacedName{
				Name:      uniqueName,
				Namespace: "default",
			}

			dep := &appsv1.Deployment{}
			err := k8sClient.Get(ctx, namespacedName, dep)
			if err != nil && errors.IsNotFound(err) {
				replicas := int32(15)
				resource := &appsv1.Deployment{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      namespacedName.Name,
						Namespace: namespacedName.Namespace,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &replicas,
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"app": "test"},
						},
						Template: v1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{"app": "test"},
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Name:  "test",
										Image: "alpine",
									},
								},
							},
						},
					},
				}

				Expect(k8sClient.Create(ctx, resource)).To(Succeed())

				resource.Status = appsv1.DeploymentStatus{
					Replicas:          replicas,
					UpdatedReplicas:   replicas,
					ReadyReplicas:     replicas,
					AvailableReplicas: replicas,
				}

				Expect(k8sClient.Status().Update(ctx, resource)).To(Succeed())
			}

			Expect(k8sClient.Get(ctx, namespacedName, dep)).To(Succeed())

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err = k8sClient.Get(ctx, namespacedName, crp)
			if err != nil && errors.IsNotFound(err) {
				maxUnavailable := int32(0)
				resource := &policyv1alpha1.ControlledReplicaPropagation{
					ObjectMeta: metav1.ObjectMeta{
						Name:      namespacedName.Name,
						Namespace: namespacedName.Namespace,
					},
					Spec: policyv1alpha1.ControlledReplicaPropagationSpec{
						Strategy: &policyv1alpha1.Strategy{
							MaxUnavailable: &maxUnavailable,
						},
						TargetRef: policyv1alpha1.CrossVersionObjectReference{
							APIVersion: dep.APIVersion,
							Kind:       dep.Kind,
							Name:       dep.Name,
						},
						Placements: []policyv1alpha1.Placement{
							{
								ClusterName: "cluster1",
								Replicas:    5,
							},
							{
								ClusterName: "cluster2",
								Replicas:    5,
							},
							{
								ClusterName: "cluster3",
								Replicas:    5,
							},
						},
					},
				}

				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}

			mustReconcileTimes(2, namespacedName)

			return namespacedName
		}

		cleanupState := func(namespacedName types.NamespacedName) {
			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err := k8sClient.Get(ctx, namespacedName, crp)
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Delete(ctx, crp)).To(Succeed())

			dep := &appsv1.Deployment{}
			err = k8sClient.Get(ctx, namespacedName, dep)
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Delete(ctx, dep)).To(Succeed())
		}

		It("should on scale down reduce Deployment's replicas and decrease PropagationPolicy's weight", func() {
			namespaceName := setupInitState(string(uuid.NewUUID()))

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			Expect(k8sClient.Get(ctx, namespaceName, crp)).To(Succeed())

			crp.Spec.Placements[0].Replicas = 3
			Expect(k8sClient.Update(ctx, crp)).To(Succeed())

			mustReconcileTimes(1, namespaceName)

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeTrue())

			mustReconcileTimes(1, namespaceName)

			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(13)))

			policy = &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeTrue())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 3,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 5,
				},
			}))

			mustReconcileTimes(1, namespaceName)

			policy = &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeFalse())

			cleanupState(namespaceName)
		})

		It("should on scale to zero reduce Deployment's replicas and remove cluster1 from PropagationPolicy", func() {
			namespaceName := setupInitState(string(uuid.NewUUID()))

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			Expect(k8sClient.Get(ctx, namespaceName, crp)).To(Succeed())

			crp.Spec.Placements = crp.Spec.Placements[1:]
			Expect(k8sClient.Update(ctx, crp)).To(Succeed())

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(10)))

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeFalse())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 5,
				},
			}))

			cleanupState(namespaceName)
		})

		It("should on scale up increase Deployment's replicas and increase PropagationPolicy's weight", func() {
			namespaceName := setupInitState(string(uuid.NewUUID()))

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			Expect(k8sClient.Get(ctx, namespaceName, crp)).To(Succeed())

			crp.Spec.Placements[0].Replicas = 50
			Expect(k8sClient.Update(ctx, crp)).To(Succeed())

			mustReconcileTimes(1, namespaceName)

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeTrue())

			mustReconcileTimes(1, namespaceName)

			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(60)))

			policy = &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeTrue())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 50,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 5,
				},
			}))

			mustReconcileTimes(1, namespaceName)

			policy = &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeFalse())

			cleanupState(namespaceName)
		})

		It("should on scale from zero increase Deployment's replicas and add cluster4 to PropagationPolicy", func() {
			namespaceName := setupInitState(string(uuid.NewUUID()))

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			Expect(k8sClient.Get(ctx, namespaceName, crp)).To(Succeed())

			newPlacements := crp.Spec.Placements
			newPlacements = append(newPlacements, policyv1alpha1.Placement{
				ClusterName: "cluster4",
				Replicas:    5,
			})
			crp.Spec.Placements = newPlacements
			Expect(k8sClient.Update(ctx, crp)).To(Succeed())

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(20)))

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeFalse())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster4"},
					},
					Weight: 5,
				},
			}))

			cleanupState(namespaceName)
		})
	})

	When("when scaling multiple clusters", func() {
		ctx := context.Background()

		setupInitState := func(uniqueName string) types.NamespacedName {
			namespacedName := types.NamespacedName{
				Name:      uniqueName,
				Namespace: "default",
			}

			dep := &appsv1.Deployment{}
			err := k8sClient.Get(ctx, namespacedName, dep)
			if err != nil && errors.IsNotFound(err) {
				replicas := int32(15)
				resource := &appsv1.Deployment{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      namespacedName.Name,
						Namespace: namespacedName.Namespace,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &replicas,
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"app": "test"},
						},
						Template: v1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{"app": "test"},
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Name:  "test",
										Image: "alpine",
									},
								},
							},
						},
					},
				}

				Expect(k8sClient.Create(ctx, resource)).To(Succeed())

				resource.Status = appsv1.DeploymentStatus{
					Replicas:          replicas,
					UpdatedReplicas:   replicas,
					ReadyReplicas:     replicas,
					AvailableReplicas: replicas,
				}

				Expect(k8sClient.Status().Update(ctx, resource)).To(Succeed())
			}

			Expect(k8sClient.Get(ctx, namespacedName, dep)).To(Succeed())

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err = k8sClient.Get(ctx, namespacedName, crp)
			if err != nil && errors.IsNotFound(err) {
				maxUnavailable := int32(0)
				resource := &policyv1alpha1.ControlledReplicaPropagation{
					ObjectMeta: metav1.ObjectMeta{
						Name:      namespacedName.Name,
						Namespace: namespacedName.Namespace,
					},
					Spec: policyv1alpha1.ControlledReplicaPropagationSpec{
						Strategy: &policyv1alpha1.Strategy{
							MaxUnavailable: &maxUnavailable,
						},
						TargetRef: policyv1alpha1.CrossVersionObjectReference{
							APIVersion: dep.APIVersion,
							Kind:       dep.Kind,
							Name:       dep.Name,
						},
						Placements: []policyv1alpha1.Placement{
							{
								ClusterName: "cluster1",
								Replicas:    5,
							},
							{
								ClusterName: "cluster2",
								Replicas:    5,
							},
							{
								ClusterName: "cluster3",
								Replicas:    5,
							},
						},
					},
				}

				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}

			mustReconcileTimes(2, namespacedName)

			return namespacedName
		}

		cleanupState := func(namespacedName types.NamespacedName) {
			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err := k8sClient.Get(ctx, namespacedName, crp)
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Delete(ctx, crp)).To(Succeed())

			dep := &appsv1.Deployment{}
			err = k8sClient.Get(ctx, namespacedName, dep)
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Delete(ctx, dep)).To(Succeed())
		}

		It("should on scale down reduce Deployment's replicas and decrease PropagationPolicy's weights", func() {
			namespaceName := setupInitState(string(uuid.NewUUID()))

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			Expect(k8sClient.Get(ctx, namespaceName, crp)).To(Succeed())

			crp.Spec.Placements = []policyv1alpha1.Placement{
				{
					ClusterName: "cluster1",
					Replicas:    2,
				},
				{
					ClusterName: "cluster2",
					Replicas:    2,
				},
			}
			Expect(k8sClient.Update(ctx, crp)).To(Succeed())

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(4)))

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeFalse())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 2,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 2,
				},
			}))

			cleanupState(namespaceName)
		})

		It("should on scale up increase Deployment's replicas and increase PropagationPolicy's weights", func() {
			namespaceName := setupInitState(string(uuid.NewUUID()))

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			Expect(k8sClient.Get(ctx, namespaceName, crp)).To(Succeed())

			crp.Spec.Placements = []policyv1alpha1.Placement{
				{
					ClusterName: "cluster1",
					Replicas:    10,
				},
				{
					ClusterName: "cluster2",
					Replicas:    20,
				},
				{
					ClusterName: "cluster3",
					Replicas:    30,
				},
				{
					ClusterName: "cluster4",
					Replicas:    40,
				},
			}
			Expect(k8sClient.Update(ctx, crp)).To(Succeed())

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(100)))

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(*policy.Spec.Suspension.Dispatching).To(BeFalse())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 10,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 20,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 30,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster4"},
					},
					Weight: 40,
				},
			}))

			cleanupState(namespaceName)
		})
	})

	When("moving replicas between clusters", func() {
		ctx := context.Background()

		setupInitState := func(uniqueName string, maxUnavailable *int32) types.NamespacedName {
			namespacedName := types.NamespacedName{
				Name:      uniqueName,
				Namespace: "default",
			}

			dep := &appsv1.Deployment{}
			err := k8sClient.Get(ctx, namespacedName, dep)
			if err != nil && errors.IsNotFound(err) {
				replicas := int32(15)
				resource := &appsv1.Deployment{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      namespacedName.Name,
						Namespace: namespacedName.Namespace,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &replicas,
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"app": "test"},
						},
						Template: v1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{"app": "test"},
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Name:  "test",
										Image: "alpine",
									},
								},
							},
						},
					},
				}

				Expect(k8sClient.Create(ctx, resource)).To(Succeed())

				resource.Status = appsv1.DeploymentStatus{
					Replicas:          replicas,
					UpdatedReplicas:   replicas,
					ReadyReplicas:     replicas,
					AvailableReplicas: replicas,
				}

				Expect(k8sClient.Status().Update(ctx, resource)).To(Succeed())
			}

			Expect(k8sClient.Get(ctx, namespacedName, dep)).To(Succeed())

			var strategy *policyv1alpha1.Strategy
			if maxUnavailable != nil {
				strategy = &policyv1alpha1.Strategy{
					MaxUnavailable: maxUnavailable,
				}
			}

			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err = k8sClient.Get(ctx, namespacedName, crp)
			if err != nil && errors.IsNotFound(err) {
				resource := &policyv1alpha1.ControlledReplicaPropagation{
					ObjectMeta: metav1.ObjectMeta{
						Name:      namespacedName.Name,
						Namespace: namespacedName.Namespace,
					},
					Spec: policyv1alpha1.ControlledReplicaPropagationSpec{
						Strategy: strategy,
						TargetRef: policyv1alpha1.CrossVersionObjectReference{
							APIVersion: dep.APIVersion,
							Kind:       dep.Kind,
							Name:       dep.Name,
						},
						Placements: []policyv1alpha1.Placement{
							{
								ClusterName: "cluster1",
								Replicas:    5,
							},
							{
								ClusterName: "cluster2",
								Replicas:    5,
							},
							{
								ClusterName: "cluster3",
								Replicas:    5,
							},
						},
					},
				}

				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}

			mustReconcileTimes(2, namespacedName)

			return namespacedName
		}

		cleanupState := func(namespacedName types.NamespacedName) {
			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			err := k8sClient.Get(ctx, namespacedName, crp)
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Delete(ctx, crp)).To(Succeed())

			dep := &appsv1.Deployment{}
			err = k8sClient.Get(ctx, namespacedName, dep)
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Delete(ctx, dep)).To(Succeed())
		}

		It("should live-migrate with max unavailable 0 strategy", func() {
			maxUnavailable := int32(0)
			namespaceName := setupInitState(string(uuid.NewUUID()), &maxUnavailable)

			// want to move 2 replicas from cluster2 to cluster3
			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			Expect(k8sClient.Get(ctx, namespaceName, crp)).To(Succeed())
			crp.Spec.Placements = []policyv1alpha1.Placement{
				{
					ClusterName: "cluster1",
					Replicas:    5,
				},
				{
					ClusterName: "cluster2",
					Replicas:    3,
				},
				{
					ClusterName: "cluster3",
					Replicas:    7,
				},
			}
			Expect(k8sClient.Update(ctx, crp)).To(Succeed())

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			// ensure scale up happened for live-migration
			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(17)))

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 7,
				},
			}))

			// simulate some new replicas become available (16/17)
			simulateAvailableReplicas(dep, 16)

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			// ensure cluster2 is not yet scaled down because new pods are not available
			dep = &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(17)))

			policy = &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 7,
				},
			}))

			// simulate all new replicas become available (17/17)
			simulateAvailableReplicas(dep, 17)

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			// desired state must be achieved
			dep = &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(15)))

			policy = &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 3,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 7,
				},
			}))

			cleanupState(namespaceName)
		})

		It("should live-migrate with max unavailable 5 strategy", func() {
			maxUnavailable := int32(2)
			namespaceName := setupInitState(string(uuid.NewUUID()), &maxUnavailable)

			// want to move 5 replicas from cluster2 to cluster3
			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			Expect(k8sClient.Get(ctx, namespaceName, crp)).To(Succeed())
			crp.Spec.Placements = []policyv1alpha1.Placement{
				{
					ClusterName: "cluster1",
					Replicas:    5,
				},
				{
					ClusterName: "cluster3",
					Replicas:    10,
				},
			}
			Expect(k8sClient.Update(ctx, crp)).To(Succeed())

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			// ensure scale up happened for live-migration
			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(20)))

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 10,
				},
			}))

			// simulate some new replicas become available (17/20)
			simulateAvailableReplicas(dep, 17)

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			// ensure cluster2 is not yet scaled down because not enough pods are yet available
			dep = &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(20)))

			policy = &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster2"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 10,
				},
			}))

			// simulate some new replicas become available (18/20)
			simulateAvailableReplicas(dep, 18)

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			// ensure the desired state is achieved because if we remove 5 from 18 available,
			// we fall to 13 available replicas which is >= desired replicas (15) - maxUnavailable (2)
			dep = &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(15)))

			policy = &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 10,
				},
			}))

			cleanupState(namespaceName)
		})

		It("should not live-migrate without strategy", func() {
			namespaceName := setupInitState(string(uuid.NewUUID()), nil)

			// want to move 5 replicas from cluster2 to cluster3
			crp := &policyv1alpha1.ControlledReplicaPropagation{}
			Expect(k8sClient.Get(ctx, namespaceName, crp)).To(Succeed())
			crp.Spec.Placements = []policyv1alpha1.Placement{
				{
					ClusterName: "cluster1",
					Replicas:    5,
				},
				{
					ClusterName: "cluster3",
					Replicas:    10,
				},
			}
			Expect(k8sClient.Update(ctx, crp)).To(Succeed())

			// suspend, update, unsuspend
			mustReconcileTimes(3, namespaceName)

			// ensure the desired state is immediately achieved, no live-migration
			dep := &appsv1.Deployment{}
			Expect(k8sClient.Get(ctx, namespaceName, dep)).To(Succeed())
			Expect(*dep.Spec.Replicas).To(Equal(int32(15)))

			policy := &karmadapolicyv1alpha1.PropagationPolicy{}
			Expect(k8sClient.Get(ctx, namespacedNamePolicy(namespaceName), policy)).To(Succeed())
			Expect(policy.Spec.Placement.ReplicaScheduling.WeightPreference.StaticWeightList).To(Equal([]karmadapolicyv1alpha1.StaticClusterWeight{
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster1"},
					},
					Weight: 5,
				},
				{
					TargetCluster: karmadapolicyv1alpha1.ClusterAffinity{
						ClusterNames: []string{"cluster3"},
					},
					Weight: 10,
				},
			}))

			cleanupState(namespaceName)
		})
	})
})

func mustReconcileTimes(reconciles int, namespacedName types.NamespacedName) {
	for range reconciles {
		controllerReconciler := &ControlledReplicaPropagationReconciler{
			Client: k8sClient,
			Scheme: k8sClient.Scheme(),
		}

		_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
			NamespacedName: namespacedName,
		})
		Expect(err).NotTo(HaveOccurred())
	}
}

func failsOnReconcile(reconciles int, namespacedName types.NamespacedName) error {
	var err error
	for i := range reconciles {
		controllerReconciler := &ControlledReplicaPropagationReconciler{
			Client: k8sClient,
			Scheme: k8sClient.Scheme(),
		}

		_, err = controllerReconciler.Reconcile(ctx, reconcile.Request{
			NamespacedName: namespacedName,
		})
		if i < reconciles-1 {
			Expect(err).NotTo(HaveOccurred())
		}
	}

	Expect(err).To(HaveOccurred())
	return err
}

func namespacedNamePolicy(crpNamespacedName types.NamespacedName) types.NamespacedName {
	return types.NamespacedName{
		Namespace: crpNamespacedName.Namespace,
		Name:      fmt.Sprintf("%s-%s", "crp-owned", crpNamespacedName.Name),
	}
}

func simulateAvailableReplicas(dep *appsv1.Deployment, available int32) {
	dep.Status.Replicas = *dep.Spec.Replicas
	dep.Status.UpdatedReplicas = *dep.Spec.Replicas
	dep.Status.ReadyReplicas = available
	dep.Status.AvailableReplicas = available

	Expect(k8sClient.Status().Update(ctx, dep)).To(Succeed())
}
