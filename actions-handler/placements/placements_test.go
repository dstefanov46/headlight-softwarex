package placements

import (
	"bytes"
	"encoding/json"
	"errors"
	"github.com/spf13/afero"
	"github.com/stretchr/testify/assert"
	"gitlab.eclipse.org/eclipse-research-labs/nephele-project/opencall-2/headlight/controlled-replica-propagation/api/v1alpha1"
	"io"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/client-go/rest/fake"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

type CachedFakeRESTClient struct {
	client     *fake.RESTClient
	cache      []v1alpha1.ControlledReplicaPropagation
	patchCount int
}

func NewFake(t *testing.T) *CachedFakeRESTClient {
	cfr := &CachedFakeRESTClient{
		cache: []v1alpha1.ControlledReplicaPropagation{},
	}

	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)

	gv := schema.GroupVersion{Group: "policy.nephele-project.eu", Version: "v1alpha1"}
	scheme := runtime.NewScheme()
	codes := serializer.NewCodecFactory(scheme)

	cfr.client = &fake.RESTClient{
		GroupVersion:         gv,
		NegotiatedSerializer: codes.WithoutConversion(),
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			path := strings.TrimLeft(req.URL.Path, "/")
			pathSplit := strings.Split(path, "/")

			switch {
			case req.Method == http.MethodGet && len(pathSplit) == 1 && pathSplit[0] == "controlledreplicapropagations":
				data := v1alpha1.ControlledReplicaPropagationList{
					TypeMeta: v1.TypeMeta{},
					ListMeta: v1.ListMeta{},
					Items:    cfr.cache,
				}
				jsonBody, err := json.Marshal(data)
				assert.NoError(t, err)

				return &http.Response{StatusCode: http.StatusOK, Header: header, Body: io.NopCloser(bytes.NewReader(jsonBody))}, nil

			case req.Method == http.MethodPatch && len(pathSplit) == 4 && pathSplit[0] == "namespaces" && pathSplit[2] == "controlledreplicapropagations":
				namespace := pathSplit[1]
				name := pathSplit[3]
				cfr.patchCount++

				patch, err := io.ReadAll(req.Body)
				assert.NoError(t, err)

				for i := 0; i < len(cfr.cache); i++ {
					if cfr.cache[i].Name == name && cfr.cache[i].Namespace == namespace {
						original := cfr.cache[i]
						originalJson, _ := json.Marshal(original)

						newJson, err := strategicpatch.StrategicMergePatch(originalJson, patch, &v1alpha1.ControlledReplicaPropagation{})
						assert.NoError(t, err)

						var newCRP v1alpha1.ControlledReplicaPropagation
						err = json.Unmarshal(newJson, &newCRP)
						assert.NoError(t, err)

						cfr.cache[i] = newCRP

						return &http.Response{StatusCode: http.StatusOK, Header: header, Body: nil}, nil
					}
				}

				return &http.Response{StatusCode: http.StatusNotFound, Header: header, Body: io.NopCloser(bytes.NewBufferString("not found"))}, nil
			}

			panic("unexpected request")
		}),
	}

	return cfr
}

func (cfr *CachedFakeRESTClient) AddCRP(crp v1alpha1.ControlledReplicaPropagation) {
	cfr.cache = append(cfr.cache, crp)
}

func (cfr *CachedFakeRESTClient) GetCRPs() []v1alpha1.ControlledReplicaPropagation {
	return cfr.cache
}

func (cfr *CachedFakeRESTClient) GetCRP(name string, namespace string) (v1alpha1.ControlledReplicaPropagation, error) {
	for _, crp := range cfr.cache {
		if crp.Name == name && crp.Namespace == namespace {
			return crp, nil
		}
	}
	return v1alpha1.ControlledReplicaPropagation{}, errors.New("not found")
}

func (cfr *CachedFakeRESTClient) PatchCount() int {
	return cfr.patchCount
}

func Test_batch_and_update(t *testing.T) {
	mutex := &sync.Mutex{}

	memFs := afero.NewMemMapFs()
	batchDir := "/tmp/batch-placements"

	err := memFs.Mkdir(batchDir, 0755)
	assert.NoError(t, err)

	fakeCachedClient := NewFake(t)

	placementsService := New(mutex, fakeCachedClient.client, memFs, batchDir)

	// given
	fakeCachedClient.AddCRP(v1alpha1.ControlledReplicaPropagation{
		TypeMeta: v1.TypeMeta{},
		ObjectMeta: v1.ObjectMeta{
			Name:      "test-1",
			Namespace: "default",
		},
		Spec: v1alpha1.ControlledReplicaPropagationSpec{
			Strategy: nil,
			TargetRef: v1alpha1.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-1",
			},
			Placements: []v1alpha1.Placement{},
		},
		Status: v1alpha1.ControlledReplicaPropagationStatus{},
	})
	fakeCachedClient.AddCRP(v1alpha1.ControlledReplicaPropagation{
		TypeMeta: v1.TypeMeta{},
		ObjectMeta: v1.ObjectMeta{
			Name:      "test-2",
			Namespace: "default",
		},
		Spec: v1alpha1.ControlledReplicaPropagationSpec{
			Strategy: nil,
			TargetRef: v1alpha1.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-2",
			},
			Placements: []v1alpha1.Placement{
				{
					ClusterName: "test-cluster-1",
					Replicas:    10,
				},
			},
		},
		Status: v1alpha1.ControlledReplicaPropagationStatus{},
	})

	// when placements are batched
	err = placementsService.AddToBatch([]Placement{
		{
			ClusterName: "test-cluster-1",
			Workload: Workload{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Namespace:  "default",
				Name:       "test-1",
			},
			Replicas: 100,
		},
		{
			ClusterName: "test-cluster-2",
			Workload: Workload{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Namespace:  "default",
				Name:       "test-1",
			},
			Replicas: 200,
		},
		{
			ClusterName: "test-cluster-1",
			Workload: Workload{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Namespace:  "default",
				Name:       "test-2",
			},
			Replicas: 100,
		},
	})
	assert.NoError(t, err)

	// when duplicate placements are batched
	time.Sleep(1000 * time.Millisecond) // simulate a later request
	err = placementsService.AddToBatch([]Placement{
		{
			ClusterName: "test-cluster-1",
			Workload: Workload{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Namespace:  "default",
				Name:       "test-1",
			},
			Replicas: 300,
		},
	})
	assert.NoError(t, err)

	// when duplicate placements with replicas 0 are batched
	time.Sleep(1000 * time.Millisecond) // simulate a later request
	err = placementsService.AddToBatch([]Placement{
		{
			ClusterName: "test-cluster-2",
			Workload: Workload{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Namespace:  "default",
				Name:       "test-1",
			},
			Replicas: 0,
		},
	})
	assert.NoError(t, err)

	// first loop updates, second loop ensures no changes to submit
	for range 2 {
		// when placements are processed
		placementsService.SubmitBatch()

		// assert that
		crp1, err := fakeCachedClient.GetCRP("test-1", "default")
		assert.NoError(t, err)
		assert.Equal(t, 1, len(crp1.Spec.Placements))
		assert.Equal(t, "test-cluster-1", crp1.Spec.Placements[0].ClusterName)
		assert.Equal(t, int32(300), crp1.Spec.Placements[0].Replicas)

		crp2, err := fakeCachedClient.GetCRP("test-2", "default")
		assert.NoError(t, err)
		assert.Equal(t, 1, len(crp2.Spec.Placements))
		assert.Equal(t, "test-cluster-1", crp1.Spec.Placements[0].ClusterName)
		assert.Equal(t, int32(100), crp2.Spec.Placements[0].Replicas)
	}
}

func Test_submit_batch_skips_noop_crp_update(t *testing.T) {
	mutex := &sync.Mutex{}
	memFs := afero.NewMemMapFs()
	batchDir := "/tmp/batch-placements"
	err := memFs.Mkdir(batchDir, 0755)
	assert.NoError(t, err)

	fakeCachedClient := NewFake(t)
	placementsService := New(mutex, fakeCachedClient.client, memFs, batchDir)
	fakeCachedClient.AddCRP(v1alpha1.ControlledReplicaPropagation{
		TypeMeta: v1.TypeMeta{},
		ObjectMeta: v1.ObjectMeta{
			Name:      "test-1",
			Namespace: "default",
		},
		Spec: v1alpha1.ControlledReplicaPropagationSpec{
			TargetRef: v1alpha1.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-1",
			},
			Placements: []v1alpha1.Placement{
				{
					ClusterName: "test-cluster-1",
					Replicas:    1,
				},
			},
		},
	})

	err = placementsService.AddToBatch([]Placement{
		{
			ClusterName: "test-cluster-1",
			Workload: Workload{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Namespace:  "default",
				Name:       "test-1",
			},
			Replicas: 1,
		},
	})
	assert.NoError(t, err)

	placementsService.SubmitBatch()

	assert.Equal(t, 0, fakeCachedClient.PatchCount())
}
