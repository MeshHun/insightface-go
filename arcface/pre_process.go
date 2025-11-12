package arcface

import (
	"fmt"
	"image"
	"image/color"

	"github.com/disintegration/imaging"
)



func transposeRGB(rgbs []float32) []float32 {
	out := make([]float32, len(rgbs))
	channelLength := len(rgbs) / 3
	for i := 0; i < channelLength; i++ {
		out[i] = rgbs[i*3]
		out[i+channelLength] = rgbs[i*3+1]
		out[i+channelLength*2] = rgbs[i*3+2]
	}
	return out
}

// pre-process input data for face-detect model
func preprocessImage(src image.Image, inputSize int ,data []float32) (float32, error) {
	channelSize := inputSize * inputSize       // 单通道像素数（H*W）
	// 检查目标切片长度是否符合要求（3*H*W）
	requiredLen := inputSize * inputSize * 3
    if len(data) != requiredLen {
        return 0, fmt.Errorf("invalid data length: expected %d, got %d", requiredLen, len(data))
    }
	// 分割通道切片（直接划分 CHW 格式的内存区域）
    redChan := data[:channelSize]              // R 通道（CHW 的第一个通道）
    greenChan := data[channelSize : 2*channelSize] // G 通道（第二个通道）
    blueChan := data[2*channelSize : 3*channelSize] // B 通道（第三个通道）
	// 计算缩放尺寸并调整图像大小
	var newHeight, newWidth int
	im_ratio := float32(src.Bounds().Dx()) / float32(src.Bounds().Dy())
	if im_ratio > 1 { // width > height
		newWidth = inputSize
		newHeight = int(float32(newWidth) / im_ratio)
	} else {
		newHeight = inputSize
		newWidth = int(float32(newHeight) * im_ratio)		
	}
	// 缩放图像并填充为正方形（补黑边）
	resized := imaging.Resize(src, newWidth, newHeight, imaging.Lanczos)
	padded := padBox(resized)

	// rgbs := make([]float32, inputSize*inputSize*3)
	// 按 CHW 格式提取像素
	for i := 0 ;i < channelSize; i++ {
		pixIdx := i * 4								// 计算当前像素在 Pix 中的起始索引（每个像素占4字节，i 是像素索引）
		r := float32(padded.Pix[pixIdx])     		// R 通道（0-255）
        g := float32(padded.Pix[pixIdx+1])   		// G 通道（0-255）
        b := float32(padded.Pix[pixIdx+2])   		// B 通道（0-255）
		redChan[i] = r
        greenChan[i] = g
        blueChan[i] = b
    }
	// 归一化处理
	for i := 0; i < channelSize; i++ {
		redChan[i] = normalize(redChan[i], 127.5, 128.0)
		greenChan[i] = normalize(greenChan[i], 127.5, 128.0)
		blueChan[i] = normalize(blueChan[i], 127.5, 128.0)
	}
	return  float32(newHeight) / float32(src.Bounds().Dy()),nil
}


func normalize(in float32, m float32, s float32) float32 {
	return (in - m) / s
}


// change image to square rect, padding with color Black
func padBox(src image.Image) *image.NRGBA {
	var maxW int

	if src.Bounds().Dx() > src.Bounds().Dy() {
		maxW = src.Bounds().Dx()
	} else {
		maxW = src.Bounds().Dy()
	}

	dst := imaging.New(maxW, maxW, color.Black)
	dst = imaging.Paste(dst, src, image.Point{0,0})

	//_ = imaging.Save(dst, "data/pading.jpg")

	return dst
}

// pre-process input data for face-features model
func preprocessFace(src image.Image, inputSize int, data []float32) error {
	// 补边处理为正方形
	result := padBox(src)

	// 校验目标切片长度
	channelSize := inputSize * inputSize
	requiredLen := 3 * channelSize             // 三通道总长度
    if len(data) != requiredLen {
        return fmt.Errorf("invalid data length: expected %d, got %d", requiredLen, len(data))
    }
	// 3. 分割通道切片（直接构建 CHW 格式的内存布局）
    redChan := data[:channelSize]              		// R 通道（CHW 第一个通道）
    greenChan := data[channelSize : 2*channelSize]  // G 通道（第二个通道）
    blueChan := data[2*channelSize : 3*channelSize] // B 通道（第三个通道）
	
	// rgbs := make([]float32, inputSize*inputSize*3)

	// j := 0
	// for i := range result.Pix {
	// 	if (i+1)%4 != 0 {
	// 		rgbs[j] = float32(result.Pix[i])
	// 		j++
	// 	}
	// }

	// 
	for i := 0; i < channelSize; i++ {
		pixIdx := i * 4 // 每个像素占 4 字节，计算当前像素的起始索引
		// 提取 R、G、B 通道值（跳过 A通道）
		r := float32(result.Pix[pixIdx])     // R 通道（0-255）
		g := float32(result.Pix[pixIdx+1])   // G 通道（0-255）
		b := float32(result.Pix[pixIdx+2])   // B 通道（0-255）

		// 直接写入对应通道（CHW 格式：按像素顺序填充每个通道）
		redChan[i] = r
		greenChan[i] = g
		blueChan[i] = b
    }

	// rgbs = transposeRGB(rgbs)

	// channelLength := len(rgbs) / 3
	// 归一化
	for i := 0; i < channelSize; i++ {
		redChan[i] = normalize(redChan[i], 127.5, 127.5)
		greenChan[i] = normalize(greenChan[i], 127.5, 127.5)
		blueChan[i] = normalize(blueChan[i], 127.5, 127.5)
	}

	return nil
}
