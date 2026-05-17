package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"harryxu.ca/gonet/data"
)

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
)

// LoadImages loads the MNIST image data file. This file consists of a
// fixed-size metadata header followed by the raw byte data for the images.
// Returns the flat normalized image data and the number of images.
func LoadImages(path string) (out []float64, n int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()

	var h struct{ Magic, N, Rows, Cols uint32 }
	if err := binary.Read(f, binary.BigEndian, &h); err != nil {
		return nil, 0, err
	}

	if h.Magic != imageMagic {
		return nil, 0, fmt.Errorf("mnist: bad images magic %#x", h.Magic)
	}

	px := int(h.Rows * h.Cols)

	rawData := make([]byte, int(h.N)*px)
	if _, err := io.ReadFull(f, rawData); err != nil {
		return nil, 0, err
	}

	out = make([]float64, len(rawData))
	for i, b := range rawData {
		// normalize data
		out[i] = float64(b) / 255.0
	}

	return out, int(h.N), nil
}

// LoadLabels loads the MNIST label data file. This file consists of a
// fixed-size metadata header followed by the raw byte data for each
// class label. Returns the flat class labels.
func LoadLabels(path string) (out []int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()

	var h struct{ Magic, N uint32 }
	if err := binary.Read(f, binary.BigEndian, &h); err != nil {
		return nil, err
	}

	if h.Magic != labelMagic {
		return nil, fmt.Errorf("mnist: bad labels magic %#x", h.Magic)
	}

	rawData := make([]byte, h.N)
	if _, err := io.ReadFull(f, rawData); err != nil {
		return nil, err
	}

	out = make([]int, h.N)
	for i, b := range rawData {
		out[i] = int(b)
	}

	return out, nil
}

// BuildSamples pairs a flat image buffer with parallel labels into a
// []data.Sample suitable for data.NewSliceDataset. Each sample's X is a
// borrowed slice view into flat (no copy), so flat must outlive the
// samples and must not be mutated while they are in use. flat must be
// exactly len(labels)*dim long.
func BuildSamples(flat []float64, labels []int, dim int) []data.Sample {
	if len(flat) != len(labels)*dim {
		panic(fmt.Sprintf("mnist: BuildSamples requires len(labels)*dim=%d*%d=%d, got len(flat)=%d", len(labels), dim, len(labels)*dim, len(flat)))
	}

	samples := make([]data.Sample, len(labels))
	for i := range samples {
		samples[i] = data.Sample{
			X: flat[i*dim : (i+1)*dim],
			Y: labels[i],
		}
	}

	return samples
}
