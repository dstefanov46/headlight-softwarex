package placements

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/google/uuid"
	"github.com/spf13/afero"
	"gitlab.eclipse.org/eclipse-research-labs/nephele-project/opencall-2/headlight/controlled-replica-propagation/api/v1alpha1"
	"k8s.io/client-go/rest"
	"log/slog"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

type Service struct {
	mutex      *sync.Mutex
	restClient rest.Interface
	fs         afero.Fs
	batchDir   string
}

func New(mutex *sync.Mutex, restClient rest.Interface, fs afero.Fs, batchDir string) *Service {
	return &Service{
		mutex,
		restClient,
		fs,
		batchDir,
	}
}

func (s *Service) AddToBatch(placements []Placement) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	timestamp := time.Now().Unix()
	for i := 0; i < len(placements); i++ {
		placements[i].Timestamp = &timestamp
	}

	requestId := uuid.New()
	filename := fmt.Sprintf("request_%s.json", requestId)
	path := filepath.Join(s.batchDir, filename)

	// Marshal the data into JSON and write to the file
	jsonData, err := json.Marshal(placements)
	if err != nil {
		return fmt.Errorf("failed to marshal data: %w", err)
	}

	err = afero.WriteFile(s.fs, path, jsonData, 0644)
	if err != nil {
		return fmt.Errorf("failed to write data to file: %w", err)
	}

	return nil
}

func (s *Service) SubmitBatch() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	files, err := afero.ReadDir(s.fs, s.batchDir)
	if err != nil {
		slog.Error("Failed to read batch placements directory: ", s.batchDir, err)
		return
	}

	batchedPlacements := make([]Placement, 0)
	filesToCleanup := make([]string, 0)
	for _, f := range files {
		path := filepath.Join(s.batchDir, f.Name())

		data, err := afero.ReadFile(s.fs, path)
		if err != nil {
			slog.Error("Failed to read file %s: %v", f.Name(), err)
			continue
		}

		var placements []Placement
		err = json.Unmarshal(data, &placements)
		if err != nil {
			slog.Error("Failed to unmarshal file %s: %v", f.Name(), err)
		}

		batchedPlacements = append(batchedPlacements, placements...)
		filesToCleanup = append(filesToCleanup, path)
	}

	if len(batchedPlacements) == 0 {
		return
	}

	batchedPlacements = compressPlacements(batchedPlacements)
	slog.Info(fmt.Sprintf("Received %d placement updates", len(batchedPlacements)))

	s.updateControlledReplicaPropagations(batchedPlacements)

	for _, f := range filesToCleanup {
		err := s.fs.Remove(f)
		if err != nil {
			slog.Error("Failed to remove file %s: %v", f, err)
		}
	}
}

func (s *Service) updateControlledReplicaPropagations(placementUpdates []Placement) {
	data, err := s.restClient.Get().Resource("controlledreplicapropagations").DoRaw(context.TODO())
	if err != nil {
		slog.Error("Failed to get CRPs: ", "err", err)
		return
	}

	crpList := &v1alpha1.ControlledReplicaPropagationList{}
	err = json.Unmarshal(data, crpList)
	if err != nil {
		slog.Error("Failed to unmarshal CRPs: ", "err", err)
		return
	}

	workloadToCRPMap := make(map[string]*v1alpha1.ControlledReplicaPropagation)
	for _, crp := range crpList.Items {
		workloadId := getCrpTargetWorkloadId(&crp)

		workloadToCRPMap[workloadId] = &crp
	}

	updatedCrps := make([]*v1alpha1.ControlledReplicaPropagation, 0)

	for _, update := range placementUpdates {
		workloadId := update.Workload.GetId()
		correspondingCRP, ok := workloadToCRPMap[workloadId]
		if !ok {
			slog.Error(fmt.Sprintf(
				"Failed to find CRP targeting %s %s in namespace %s",
				update.Workload.Kind,
				update.Workload.Name,
				update.Workload.Namespace),
			)
			continue
		}

		x := -1
		for i := 0; i < len(correspondingCRP.Spec.Placements); i++ {
			if correspondingCRP.Spec.Placements[i].ClusterName == update.ClusterName {
				x = i
				break
			}
		}

		hasUpdated := false
		if x == -1 {
			if update.Replicas > 0 {
				// add new placement
				correspondingCRP.Spec.Placements = append(correspondingCRP.Spec.Placements, v1alpha1.Placement{
					ClusterName: update.ClusterName,
					Replicas:    update.Replicas,
				})
				hasUpdated = true
			}
		} else {
			if update.Replicas > 0 {
				// update placement
				if correspondingCRP.Spec.Placements[x].Replicas != update.Replicas {
					correspondingCRP.Spec.Placements[x].Replicas = update.Replicas
					hasUpdated = true
				}
			} else {
				// remove placement
				correspondingCRP.Spec.Placements = append(correspondingCRP.Spec.Placements[:x], correspondingCRP.Spec.Placements[x+1:]...)
				hasUpdated = true
			}
		}

		if hasUpdated {
			sortPlacements(correspondingCRP.Spec.Placements)
			if !isCRPInList(updatedCrps, correspondingCRP) {
				updatedCrps = append(updatedCrps, correspondingCRP)
			}
		}
	}

	for _, crp := range updatedCrps {
		patch := map[string]interface{}{
			"spec": map[string]interface{}{
				"placements": crp.Spec.Placements,
			},
		}
		patchBytes, err := json.Marshal(patch)
		if err != nil {
			slog.Error("Failed to marshal patch payload: ", "err", err)
			continue
		}

		// todo: retry?
		do := s.restClient.Patch("application/merge-patch+json").
			Resource("controlledreplicapropagations").
			Name(crp.Name).
			Namespace(crp.Namespace).
			Body(patchBytes).
			Do(context.TODO())
		if do.Error() != nil {
			slog.Error(fmt.Sprintf("Failed to update CRP %s in namespace %s", crp.Name, crp.Namespace), "err", do.Error())
		}

		slog.Info(fmt.Sprintf("Updated CRP %s in namespace %s", crp.Name, crp.Namespace))
	}
}

func isCRPInList(crpList []*v1alpha1.ControlledReplicaPropagation, crp *v1alpha1.ControlledReplicaPropagation) bool {
	for _, c := range crpList {
		if c.Name == crp.Name && c.Namespace == crp.Namespace {
			return true
		}
	}

	return false
}

func sortPlacements(placements []v1alpha1.Placement) {
	sort.Slice(placements, func(i, j int) bool {
		return placements[i].ClusterName < placements[j].ClusterName
	})
}

func getCrpTargetWorkloadId(crp *v1alpha1.ControlledReplicaPropagation) string {
	return fmt.Sprintf("%s-%s-%s-%s",
		crp.Spec.TargetRef.APIVersion,
		crp.Spec.TargetRef.Kind,
		crp.Namespace,
		crp.Spec.TargetRef.Name,
	)
}

// If there are duplicate placements, we keep only the latest one
// based on the timestamp.
func compressPlacements(placements []Placement) []Placement {
	uniquePlacements := make(map[string]Placement)

	for _, p := range placements {
		// Create a key based on Workload and ClusterName
		key := fmt.Sprintf(
			"%s-%s-%s-%s-%s",
			p.ClusterName,
			p.Workload.APIVersion,
			p.Workload.Kind,
			p.Workload.Namespace,
			p.Workload.Name,
		)

		if existing, exists := uniquePlacements[key]; !exists || *p.Timestamp > *existing.Timestamp {
			uniquePlacements[key] = p
		}
	}

	res := make([]Placement, 0, len(uniquePlacements))
	for _, p := range uniquePlacements {
		res = append(res, p)
	}

	return res
}
