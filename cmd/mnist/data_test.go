package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestLoadImages tests that LoadImages parses a synthetic IDX images
// blob and normalizes each pixel byte to a float64 in [0, 1].
func TestLoadImages(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tiny-images")

	// Build a synthetic IDX: magic=0x00000803, N=1, rows=2, cols=2, then 4 bytes.
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.BigEndian, uint32(0x00000803))
	binary.Write(buf, binary.BigEndian, uint32(1))
	binary.Write(buf, binary.BigEndian, uint32(2))
	binary.Write(buf, binary.BigEndian, uint32(2))
	buf.Write([]byte{0, 127, 255, 64})
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	flat, n, err := LoadImages(path)
	if err != nil {
		t.Fatalf("LoadImages: %v", err)
	}

	if got, want := n, 1; got != want {
		t.Errorf("LoadImages n = %d, want %d", got, want)
	}

	if got, want := len(flat), 4; got != want {
		t.Errorf("LoadImages len(flat) = %d, want %d", got, want)
	}

	if got, want := flat[0], 0.0; got != want {
		t.Errorf("LoadImages flat[0] = %v, want %v (byte 0 normalized)", got, want)
	}

	if got, want := flat[2], 1.0; got != want {
		t.Errorf("LoadImages flat[2] = %v, want %v (byte 255 normalized)", got, want)
	}
}

// TestLoadLabels tests that LoadLabels parses a synthetic IDX labels
// blob and returns the bytes as []int in order.
func TestLoadLabels(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tiny-labels")

	// Build a synthetic IDX: magic=0x00000801, N=3, then 3 label bytes.
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.BigEndian, uint32(0x00000801))
	binary.Write(buf, binary.BigEndian, uint32(3))
	buf.Write([]byte{5, 0, 9})
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	labels, err := LoadLabels(path)
	if err != nil {
		t.Fatalf("LoadLabels: %v", err)
	}

	want := []int{5, 0, 9}
	if !reflect.DeepEqual(labels, want) {
		t.Errorf("LoadLabels = %v, want %v", labels, want)
	}
}

// TestLoadImagesBadMagic tests that LoadImages refuses a file whose
// magic number doesn't match the IDX image format.
func TestLoadImagesBadMagic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "wrong-magic")

	buf := new(bytes.Buffer)
	binary.Write(buf, binary.BigEndian, uint32(0x00000801)) // labels magic
	binary.Write(buf, binary.BigEndian, uint32(1))
	binary.Write(buf, binary.BigEndian, uint32(2))
	binary.Write(buf, binary.BigEndian, uint32(2))
	buf.Write([]byte{0, 0, 0, 0})
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	if _, _, err := LoadImages(path); err == nil {
		t.Errorf("LoadImages with bad magic: expected error, got nil")
	}
}

// TestBuildSamples tests that samples are paired correctly
// and X is a borrowed view into flat (not a copy).
func TestBuildSamples(t *testing.T) {
	flat := []float64{1, 2, 3, 4, 5, 6}
	labels := []int{7, 8}
	dim := 3

	samples := BuildSamples(flat, labels, dim)

	if got, want := len(samples), 2; got != want {
		t.Errorf("BuildSamples len = %d, want %d", got, want)
	}

	wantX := [][]float64{{1, 2, 3}, {4, 5, 6}}
	wantY := []int{7, 8}
	for i, s := range samples {
		if !reflect.DeepEqual(s.X, wantX[i]) {
			t.Errorf("samples[%d].X = %v, want %v", i, s.X, wantX[i])
		}
		if s.Y != wantY[i] {
			t.Errorf("samples[%d].Y = %d, want %d", i, s.Y, wantY[i])
		}
	}

	// Mutating flat should be visible through samples; pins the
	// no-copy borrowing contract documented on BuildSamples.
	flat[0] = 99
	if got, want := samples[0].X[0], 99.0; got != want {
		t.Errorf("after mutating flat, samples[0].X[0] = %v, want %v (X should borrow, not copy)", got, want)
	}
}

// TestBuildSamplesLengthMismatch tests that BuildSamples panics when
// flat isn't exactly len(labels)*dim long, catching mismatched inputs
// at the call site instead of letting them produce silently wrong
// training batches.
func TestBuildSamplesLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("BuildSamples with mismatched length: expected panic, got none")
		}
	}()

	flat := []float64{1, 2, 3, 4, 5} // 5 elements; labels(2) * dim(3) = 6
	labels := []int{7, 8}
	BuildSamples(flat, labels, 3)
}
