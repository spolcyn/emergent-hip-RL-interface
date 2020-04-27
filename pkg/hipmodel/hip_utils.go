package hipmodel

import (
	//"log"
	"math"
	"strconv"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
)

// SP: Test Slice Equality
func SlicesAreEqual(s1, s2 []int) bool {

	if len(s1) != len(s2) {
		return false
	}

	for i := 0; i < len(s1); i++ {
		if s1[i] != s2[i] {
			return false
		}
	}

	return true
}

// SP: Finds the most similar vocabulary pattern to a provided activation set within a vocabulary
func FindMostSimilarVocabPattern(pattern *etensor.Float32, vocab patgen.Vocab) *etensor.Float32 {

	distance := math.MaxInt32
	var mostSimilar *etensor.Float32

	// iterate through all vocab items, storing the one with the lowest distance from our pattern
	for _, vItem := range vocab {
		tmpDistance := CalculateDistance(pattern, vItem)

		if tmpDistance < distance {
			distance = tmpDistance
			mostSimilar = vItem
		}
	}

	return mostSimilar.Clone().(*etensor.Float32)

}

type NameError struct {
	Distance       int
	ClosestPattern *etensor.Float32
}

// SP: Given a candidate and a reference, outputs a distance score between them
// Currently uses Hamming Distance
// Tensors are float-types, so can't easily use XOR as one could on a bit array
// On our size tensors, shouldn't matter -- if tensors get too big, also easily parallelizable
func CalculateDistance(t1, t2 *etensor.Float32) int {

	// TODO: Throw an error instead
	if !SlicesAreEqual(t1.ShapeObj().Shp, t2.ShapeObj().Shp) {
		panic("tensors don't have the same shape!")
	}

	distance := 0

	for i := 0; i < t1.ShapeObj().Len(); i++ {
		if math.Abs(math.Round(float64(t1.Value1D(i)))) != math.Abs(math.Round(float64(t2.Value1D(i)))) {
			distance++
		}
	}

	return distance
}

// SP: Calculates which vocabulary pattern the ECOut is most similar to,
// then compares that pattern to the ECIn to see if it is correct or not
// We return the distance between the patterns for downstream functions
// to make decisions about what it means
func (ss *Sim) CalculateError(ecin *leabra.Layer, ecout *leabra.Layer) {

	// get activations
	outPattern := etensor.NewFloat32(ecout.Shape().Shp, nil, nil)
	ecout.UnitValsTensor(outPattern, "Act")

	//log.Printf("Outpattern:\n\n%v", outPattern)

	distance := math.MaxInt32
	var mostSimilar *etensor.Float32

	// find closest pattern in the dataset to the output pattern
	for i := 0; i < ss.TestAB.NumRows(); i++ {

		tmpPattern := ss.TestAB.ColByName("Input").SubSpace([]int{0}).(*etensor.Float32)
		tmpDistance := CalculateDistance(outPattern, tmpPattern)

		if tmpDistance < distance {
			distance = tmpDistance
			mostSimilar = tmpPattern
		}
	}

	// set the error pattern
	ss.NameErrorResult = &NameError{Distance: distance, ClosestPattern: mostSimilar.Clone().(*etensor.Float32)}
}

// updates the testenv with a pattern so that the index 0 of the testenv will return the test pattern
// for use with testing a very specific, arbitrary pattern that can't be pre-loaded in a dataset
// because we have no idea what the cortex model is going to send at us
func (ss *Sim) UpdateEnvWithTestPattern(tsr *etensor.Float32) {

	var env env.FixedTable = ss.TestEnv

	// create new etable with pattern
	table := etable.NewTable("TestPattern")
	table.AddCol(tsr, "Input")
	table.AddCol(tsr, "ECout")

	// create index view from etable for TestEnv.Table
	env.Table = etable.NewIdxView(table)

	// set TestEnv Sequential
	env.Sequential = true

	// Valdiate TestEnv
	env.Validate()

	// Init TestEnv
	env.Init(0)

}

func SaveVocabToCSV(filename string, vocab patgen.Vocab) {

	/* save to CSV */
	table := etable.NewTable(filename)

	numUnits := 1
	table.AddRows(numUnits)

	for k, v := range vocab {
		table.AddCol(v, k)
	}

	table.SaveCSV(gi.FileName(filename), etable.Comma, true)
}

// loads a pre-created vocab from vocabFilename, creating the vocab if it doesn't exist
// the dataset is always re-created to avoid having to create a scheme to verify whether a dataset matches a given vocab
func LoadDataset(vocabFilename string) *etable.Table {

	dataset := etable.NewTable("TrainAB")

	CreateDataset(dataset)
	dataset.SaveCSV(gi.FileName("dataset.csv"), etable.Comma, true)

	return dataset
}

/* create a dataset in the provided table from the provided vocab */
func CreateDataset(dt *etable.Table) {

	npats := 20
	ySize := 6
	xSize := 2
	poolY := 3
	poolX := 4

	// creates the etable of the approrpiate size
	// Tensor Format: <ySize, xSize, poolY, poolX>
	patgen.InitPats(dt, "TrainAB-Me", "Vocab Generated Patterns", "Input", "ECout", npats, ySize, xSize, poolY, poolX)

	// configure vocab parameters
	vocab := patgen.Vocab{}
	var pctAct float32 = .25 // this might be too high -- .2 in hip_bench, .25 in OG train_ab.tsv
	var minDiff float32 = .4
	vocabNames := make([]string, ySize*xSize)

	// create the vocab items
	for i := 0; i < ySize*xSize; i++ {
		vocabNames[i] = strconv.Itoa(i)
		patgen.AddVocabPermutedBinary(vocab, vocabNames[i], npats, poolY, poolX, pctAct, minDiff)
	}

	// mix input and output patterns into training dataset
	patgen.MixPats(dt, vocab, "Input", vocabNames)
	patgen.MixPats(dt, vocab, "ECout", vocabNames)
}
