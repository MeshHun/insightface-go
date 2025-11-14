package arcface

import (
	"image"
)

// 人脸信息结构体，包含检测框、关键点、置信度、特征向量
type Face struct {
	BoundingBox [4]int     // [x1, y1, x2, y2]
	Landmarks   [10]float32 // 5个关键点，每个点(x,y)，共10个值
	Score       float32    // 检测置信度
	Features    []float32  // 特征向量（512维）
}

// 引擎配置参数
type Config struct {
	ModelPath      string  // 模型文件夹路径（如"./buffalo_l"）
	DetectThreshold float32 // 检测置信度阈值（默认0.5）
	NMSThreshold   float32 // NMS阈值（默认0.4）
}

// 人脸引擎结构体，管理模型会话和配置
type FaceEngine struct {
	session *ModelSessions
	config  Config
}


// 创建新的人脸引擎
func NewFaceEngine(config Config) (*FaceEngine, error) {
	// 设置默认配置
	if config.DetectThreshold == 0 {
		config.DetectThreshold = 0.5
	}
	if config.NMSThreshold == 0 {
		config.NMSThreshold = 0.4
	}
	// 加载模型
	session, err := LoadOnnxModel(config.ModelPath)
	if err != nil {
		return nil, err
	}
	// 覆盖默认阈值（如果需要）
	nms_thresh = config.NMSThreshold
	det_thresh = config.DetectThreshold

	return &FaceEngine{
		session: session,
		config:  config,
	}, nil
}

func (e *FaceEngine) Close() error {
	// 释放ONNX会话资源
	if e.session.detSession != nil {
		e.session.detSession.Destroy()
	}
	if e.session.arcSession != nil {
		e.session.arcSession.Destroy()
	}
	// 释放输入输出张量
	e.session.detInput.Destroy()
	for _, output := range e.session.detOutputs {
		output.Destroy()
	}
	e.session.arcInput.Destroy()
	e.session.arcOutput.Destroy()
	return nil
}

// 检测图片中的人脸（返回检测框、关键点、置信度）
func (e *FaceEngine) Detect(img image.Image) ([]Face, error) {
	dets, kpss, err := FaceDetect(e.session, img)
	if err != nil {
		return nil, err
	}

	var faces []Face
	for i := range dets {
		// 转换检测框为int类型
		bbox := [4]int{
			int(dets[i][0]),
			int(dets[i][1]),
			int(dets[i][2]),
			int(dets[i][3]),
		}
		// 提取关键点（前10个值，忽略分数）
		var landmarks [10]float32
		copy(landmarks[:], kpss[i][:10])

		faces = append(faces, Face{
			BoundingBox: bbox,
			Landmarks:   landmarks,
			Score:       dets[i][4], // 置信度在dets的第5个位置
		})
	}
	return faces, nil
}

// 为检测到的人脸提取特征向量
func (e *FaceEngine) ExtractFeatures(img image.Image, face *Face) error {
	// 使用关键点对齐人脸并提取特征
	features, _, err := FaceFeatures(e.session, img, face.Landmarks[:])
	if err != nil {
		return err
	}
	face.Features = features
	return nil
}

// 封装接口 检测人脸并提取所有特征
func (e *FaceEngine) get(img image.Image) ([]Face, error) {
	faces, err := e.Detect(img)
	if err != nil {
		return nil, err
	}
	// 为每个人脸提取特征
	for i := range faces {
		if err := e.ExtractFeatures(img, &faces[i]); err != nil {
			return nil, err
		}
	}
	return faces, nil
}