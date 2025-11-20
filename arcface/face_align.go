package arcface

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

var (
	// arcface matrix from insightface/utils/face_align.py
	arcface_src = []gocv.Point2f{
	   {X:38.2946, Y:51.6963},
       {X:73.5318, Y:51.5014},
       {X:56.0252, Y:71.7366},
       {X:41.5493, Y:92.3655},
       {X:70.7299, Y:92.2041},
   }
)


// Crop face image and normalization
func norm_crop(srcImage image.Image, lmk []float32) (image.Image, error) {
	// similarity transform
	m := estimate_norm(lmk)
	defer m.Close()

	// print out the 2×3 transformation matrix
	//printM(m)

	// transfer to Mat
	src, err := gocv.ImageToMatRGB(srcImage)
	if err!=nil {
		return nil, err
	}
	defer src.Close()
	dst := src.Clone()
	defer dst.Close()

	// affine transformation to an image (Mat)
	gocv.WarpAffine(src, &dst, m, image.Point{face_align_image_size, face_align_image_size})

	// Mat transfer to image
	aimg, err := dst.ToImage()
	if err!=nil {
		return nil, err
	}

	return aimg, nil
}


// equal to python: skimage.transform.SimilarityTransform()
func estimate_norm(lmk []float32) gocv.Mat {
	dst := make([]gocv.Point2f, 5)
	for i:=0;i<5;i++ {
		dst[i] = gocv.Point2f{X:lmk[i*2], Y:lmk[i*2+1]}
	}

	pvsrc := gocv.NewPoint2fVectorFromPoints(arcface_src)
	defer pvsrc.Close()

	pvdst := gocv.NewPoint2fVectorFromPoints(dst)
	defer pvdst.Close()

	inliers := gocv.NewMat()
	defer inliers.Close()
	method := 4 // cv2.LMEDS
	ransacProjThreshold := 3.0
	maxiters := uint(2000)
	confidence := 0.99
	refineIters := uint(10)

	m := gocv.EstimateAffinePartial2DWithParams(pvdst, pvsrc, inliers, method, 
												 ransacProjThreshold, maxiters, confidence, refineIters)

	return m
}

// print matrix, for test
func printM(m gocv.Mat) {
	for i:=0;i<m.Rows();i++ {
		for j:=0;j<m.Cols();j++ {
			fmt.Printf("%v ", m.GetDoubleAt(i, j))
		}
		fmt.Printf("\n")
	}
}