//Example1
package main

import (
	"arcface-go/arcface"
	"image"
	"image/color"
	"log"

	"github.com/disintegration/imaging"
	"gocv.io/x/gocv"
)

const (
	test_image_path = "diego.jpg"
)


func main() {
	session,err := arcface.LoadOnnxModel("./buffalo_l")
	if err!=nil {
		log.Fatal("Load model fail: ", err.Error())
	}

	// load image
	srcImage, err := imaging.Open(test_image_path)
	if err != nil {
		log.Fatal("Open image error: ", err.Error())
	}

	dets, kpss, err := arcface.FaceDetect(session,srcImage)
	if err != nil {
		log.Fatal("FaceDetect() error: ", err.Error())
	}

	log.Println("face num: ", len(kpss))

	if len(dets)==0 {
		log.Println("No face detected.")
		return		
	}
	// --------------- 绘制定位框 ---------------
	// 将 image.Image 转换为 gocv.Mat 以便绘图
	mat, err := gocv.ImageToMatRGB(srcImage)
	if err != nil {
		log.Fatal("Convert image to mat failed: ", err)
	}
	defer mat.Close() // 确保资源释放

	// 定义框的颜色（BGR格式，这里是红色）和线宽
	// color := gocv.NewScalar(0, 0, 255, 0) // OpenCV 使用 BGR 而非 RGB
	color := color.RGBA{R: 255, G: 0, B: 0, A: 255}
	thickness := 2

	// 遍历所有检测到的人脸框并绘制
	for _, det := range dets {
		// det 格式: [x1, y1, x2, y2, score]
		x1 := int(det[0])
		y1 := int(det[1])
		x2 := int(det[2])
		y2 := int(det[3])
		rect:= image.Rect(x1, y1, x2, y2)
		// 绘制矩形框
		gocv.Rectangle(&mat, rect, color, thickness)
	}

	// 保存绘制后的图片
	outputPath := "detected_result.jpg"
	if ok := gocv.IMWrite(outputPath, mat); !ok {
		log.Fatal("Failed to save detected image")
	}
	log.Printf("Detected result saved to %s", outputPath)
	// -----------------------------------------

	/*
	// crop face by detect boxes without normalization
	sr := image.Rectangle{
		image.Point{int(dets[0][0]), int(dets[0][1])}, 
		image.Point{int(dets[0][2]), int(dets[0][3])},
	}
	src2 := imaging.Crop(srcImage, sr)
	_ = imaging.Save(src2, "crop_face.jpg")
	*/


	// just use the first face data, which score is the highest
	features, normFace, err := arcface.FaceFeatures(session, srcImage, kpss[0])
	if err != nil {
		log.Fatalf("FaceFeatures() error: %s\n", err.Error())
	}

	// normalized face image
	_ = imaging.Save(normFace, "norm_face.jpg")

	log.Println("features: ", features)
}

// Example2
// package main

// import (
// 	"arcface-go/arcface"
// 	"log"

// 	"github.com/disintegration/imaging"
// )

// func main() {
// 	// 1. 初始化引擎
// 	config := arcface.Config{
// 		ModelPath:      "./buffalo_l",
// 		DetectThreshold: 0.6, // 提高检测阈值
// 	}
// 	engine, err := arcface.NewFaceEngine(config)
// 	if err != nil {
// 		log.Fatal("初始化引擎失败：", err)
// 	}
// 	defer engine.Close() // 确保资源释放

// 	// 2. 加载图片
// 	img, err := imaging.Open("test.jpg")
// 	if err != nil {
// 		log.Fatal("加载图片失败：", err)
// 	}

// 	// 3. 处理（检测+提取特征）
// 	faces, err := engine.Get(img)
// 	if err != nil {
// 		log.Fatal("处理图片失败：", err)
// 	}

// 	// 4. 输出结果
// 	log.Printf("检测到 %d 张人脸\n", len(faces))
// 	for i, face := range faces {
// 		log.Printf("  人脸 %d：\n", i)
// 		log.Printf("  检测框：%v\n", face.BoundingBox)
// 		log.Printf("  置信度：%.2f\n", face.Score)
// 		log.Printf("  特征向量长度：%v\n", (face.Features[:10]))     //由于使用内存指针，导致所有特征向量指向同一块内存。
// 	}
// }