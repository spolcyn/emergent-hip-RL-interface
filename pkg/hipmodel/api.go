// api.go
// implements the methods interacting between the model and the server

package hipmodel

import (
	"errors"
	"fmt"
	"log"
)

/* Update the data used for training the model */
func (ss *Sim) RestUpdateTrainingData(update *DatasetUpdate) error {
	log.Printf("updating training data with file: %v\n", update.Filename)

	err := ss.OpenPatComma(ss.TrainAB, update.Filename, "TrainAB", "AB Training Patterns")

	return err
}

/* Update the data used for comparing output patterns to for error determination */
func (ss *Sim) RestUpdateInputPatternData(update *DatasetUpdate) error {
	log.Printf("updating input patterns data with file: %v\n", update.Filename)

	// as it turns out, TestAB is where the IdxView for the TestItem comes from, so we need to update that
	// NOTE: for now, we're still opening InputPatterns as well, but long-term that shouldn't happen
	ss.OpenPatComma(ss.InputPatterns, update.Filename, "InputPatterns", "Input Patterns")

	return nil
}

/* Test an item */
func (ss *Sim) RestTestPattern(tr *TestRequest) (*NameError, error) {
	if ss.IsRunning {
		return nil, errors.New("Model is already running, couldn't test item yet")
	}

	ss.IsRunning = true
	log.Printf("testing pattern: %v\n", tr.Pattern)

	// setup the environment as we want it
	tsr := ParseTensorFromJSON(tr.Shape, tr.Pattern)
	ss.UpdateEnvWithTestPattern(tsr)

	ss.TestItem(0) // always use 0, that's where we'll put the item
	ss.IsRunning = false

	return ss.NameErrorResult, nil
}

/* Start the model's training process */
func (ss *Sim) RestStartTraining(tr *TrainRequest) (string, error) {

	/*
		action, success := ss.ToolBar.FindActionByName("Train")

		if !success {
			log.Println("Error, 'Train' button not found")
			return "", errors.New("Train GUI button not found")
		}*/

	if ss.IsRunning {
		return "", errors.New("Training is already running")
	}

	ss.MaxRuns = tr.MaxRuns
	ss.MaxEpcs = tr.MaxEpcs

	// re-init model to clear previous weights and reset training parameters
	ss.Init()

	// start the training in a goroutine
	go ss.Train()
	ss.IsRunning = true

	return fmt.Sprintf("Training started. Max Runs: %v, Max Epochs: %v\n", ss.MaxRuns, ss.MaxEpcs), nil
}

/* Check on the model's training status */
func (ss *Sim) GetTrainingStatus() string {
	if ss.IsRunning {
		return "Training"
	} else {
		return "Not Training"
	}
}
