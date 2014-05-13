package main

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

func main() {
	runtime.GOMAXPROCS(1) // Use 1 to 4 cores

	//Initialize main parameters
	//******************
	var k0 float64 = 1       // initial cake size
	var beta float64 = 0.9   // discount factor
	var kpoints int = 100    // number of different cake sizes to consider (grid size)
	var tol float64 = 0.0001 // tolerance value

	// Create grid for k (cake sizes to consider)
	//******************
	var kincr float64 = k0 / (float64(kpoints) - 1) // calculate increment such that grid size is equal to kpoints

	// 1 x kpoints row vector with elements from 0 to k0 in increments of kincr
	var cakeVector = make([]float64, kpoints)
	var i int = 1
	var incPoint float64 = 0
	for i <= kpoints {
		cakeVector[i-1] = incPoint
		incPoint = incPoint + kincr
		i = i + 1
	}
	cakeVector[0] = 6.3829e-4 //set startingpoint very smal but not zero

	//Initialize consumption matrix
	consum := make([][]float64, kpoints)
	for m := range cakeVector {
		consum[m] = make([]float64, kpoints)
	}

	for m := range cakeVector {
		k := m
		for n := 0; n < kpoints; n++ {
			if k >= 0 {
				consum[m][n] = cakeVector[k]
				k--
			} else {
				consum[m][n] = math.NaN()
			}
		}
	}
	//fmt.Println(consum)

	//Initialize utility matrix
	utility := make([][]float64, kpoints)
	for m := range cakeVector {
		utility[m] = make([]float64, kpoints)
	}
	for m := range cakeVector {
		for n := 0; n < kpoints; n++ {
			utility[m][n] = utilityFuncition(consum[m][n])
		}
	}

	// 1 x kpoints row vector of zeros, which is our initial guess for the value function V(k)
	var V = make([]float64, kpoints)
	i = 1
	for i <= kpoints {
		V[i-1] = 0
		i = i + 1
	}

	var newV = make([]float64, kpoints)

	//Initialize profit matrix
	profit := make([][]float64, kpoints)
	for m := range cakeVector {
		profit[m] = make([]float64, kpoints)
	}

	//Initialize aux matrix
	aux := make([][]float64, kpoints)
	for m := range cakeVector {
		aux[m] = make([]float64, kpoints)
	}

	//Initialize index
	index := make([]int, kpoints)

	//Define tolerance
	gap := tol + 1

	// Value function iteration
	for gap > tol {
		//apply the Bellman operator TV(k)=max{u(k,k')+beta*V(k')} until TV(k) and V(k) are close
		profit = bellmanOperation(utility, beta, V, kpoints, &aux, &profit) //kpoints x kpoints matrix, which forms the RHS of the Bellman equation
		newV, index = maxMatrix(profit, kpoints)                            //two 1 x kpoints matrices: one for maximum values, the other for indices
		gap = calGap(V, newV)
		V = newV
	}

	//At the end
	fmt.Println("optimal consumtion:")
	for i := range index {
		fmt.Println(cakeVector[index[i]])
	}
	fmt.Println(index)
}

func calGap(V []float64, newV []float64) float64 {

	var gap float64 = 0
	for index, element := range V {
		gap = gap + math.Abs((newV[index] - element))
	}
	return gap
}

func maxMatrix(matrix [][]float64, length int) ([]float64, []int) {

	//Initialize variables
	maxValue := make([]float64, length)
	maxIndex := make([]int, length)
	var wg sync.WaitGroup

	//find max
	for m := 0; m < length; m++ {
		maxValue[m] = -1000000
		maxIndex[m] = -1

		wg.Add(1)
		go maxConcurrent(&wg, matrix, length, m, &maxValue, &maxIndex)
	}

	wg.Wait()

	return maxValue, maxIndex
}

func maxConcurrent(wg *sync.WaitGroup, matrix [][]float64, length int, m int, maxValue *[]float64, maxIndex *[]int) {
	for n := 0; n < length; n++ {
		if matrix[m][n] > (*maxValue)[m] {
			(*maxValue)[m] = matrix[m][n]
			(*maxIndex)[m] = n
		}
	}
	wg.Done()
}

func bellmanOperation(utility [][]float64, beta float64, V []float64, length int, auxP *[][]float64, profitP *[][]float64) [][]float64 {

	aux := *auxP
	profit := *profitP
	var wg sync.WaitGroup

	//calc aux matrix
	for m := 0; m < length; m++ {
		wg.Add(1)
		go auxConcurrent(&wg, V, beta, m, length, &aux)
	}

	wg.Wait()

	//calc profit matrix
	for m := 0; m < length; m++ {
		wg.Add(1)
		go profitConcurrent(&wg, aux, utility, m, length, &profit)
	}

	wg.Wait()

	return profit
}

func auxConcurrent(wg *sync.WaitGroup, V []float64, beta float64, m int, length int, aux *[][]float64) {

	for n := 0; n < length; n++ {
		(*aux)[m][n] = V[n] * beta
	}

	wg.Done()
}

func profitConcurrent(wg *sync.WaitGroup, aux [][]float64, utility [][]float64, m int, length int, profit *[][]float64) {

	for n := 0; n < length; n++ {
		(*profit)[m][n] = utility[m][n] + aux[m][n]
	}

	wg.Done()
}

func utilityFuncition(value float64) float64 {

	return math.Log(value)
}
