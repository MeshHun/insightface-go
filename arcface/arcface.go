// Package to provide face-detection and features-retrieving
package arcface

import (
	"fmt"
	// "errors"
	"path/filepath"
	"image"

	ort "github.com/yalue/onnxruntime_go"
)


const (
	det_size = 640
	face_align_image_size = 112
)

type ModelSessions struct {
	detSession *ort.AdvancedSession
	arcSession *ort.AdvancedSession
	detInput   *ort.Tensor[float32]
	detOutputs  []ort.Value
	arcInput   *ort.Tensor[float32]
	arcOutput  *ort.Tensor[float32]
}

// type recModelSessions struct {
// 	arcSession *ort.AdvancedSession
// 	arcInput   *ort.Tensor[float32]
// 	arcOutput  *ort.Tensor[float32]
// }

var (
	// detSession   	 *ort.AdvancedSession
	// arcSession 		 *ort.AdvancedSession
	detInputName  = []string{"input.1"}	
	detOutputNames = []string{
		"448", "471", "494",
		"451", "474", "497",
		"454", "477", "500",
	}
	detoutputShapes = []ort.Shape{
		ort.NewShape(12800, 1),   // 443
		ort.NewShape(3200, 1),    // 468
		ort.NewShape(800, 1),     // 493
		ort.NewShape(12800, 4),   // 446
		ort.NewShape(3200, 4),    // 471
		ort.NewShape(800, 4),     // 496
		ort.NewShape(12800, 10),  // 449
		ort.NewShape(3200, 10),   // 474
		ort.NewShape(800, 10),    // 499
	}
	
	arcInputName  = []string{"input.1"}
	arcOutputName = []string{"683"}
)

// 初始化ONNX运行时环境（全局仅需调用一次）
func initORT(lib_path string) error {
	// 设置ONNX Runtime共享库路径（根据系统类型调整）
	// 例如: Windows使用"onnxruntime.dll"，Linux使用"libonnxruntime.so"
	ort.SetSharedLibraryPath(filepath.Join(lib_path))
	
	if err := ort.InitializeEnvironment(ort.WithLogLevelError()); err != nil {
		return err
	}
	return nil
}

// Load onnx model from infightface, based on "buffalo_l" (det_10g.onnx, w600k_r50.onnx).
// onnxmodel_path is the path way to onnx models.
func LoadOnnxModel(onnxmodel_path string, lib_path string) (*ModelSessions ,error) {
	if err := initORT(lib_path); err != nil {
		return nil, fmt.Errorf("error initializing ORT environment: %w", err)
	}
	detModelPath := filepath.Join(onnxmodel_path, "det_10g.onnx")
	detinputShape := ort.NewShape(1, 3, det_size, det_size)
	detInput, err := ort.NewEmptyTensor[float32](detinputShape)
	if err != nil {
		return nil, fmt.Errorf("error creating detection input tensor: %w", err)
	}
	// detoutputTensors := make([]*ort.Tensor[float32], len(detOutputNames))
	detOutputs := make([]ort.Value,len(detOutputNames))
	for i, shape := range detoutputShapes {
		detOutputs[i], err = ort.NewEmptyTensor[float32](shape)
		if err != nil {
			for j := 0; j < i; j++ {
                detOutputs[j].Destroy()
            }
			detInput.Destroy()
			return nil, fmt.Errorf("error creating detection output tensor %d: %w", i, err)
		}
		// detOutput[i] = detoutputTensors[i]
	}
	ortDetSO, err := ort.NewSessionOptions()     //ORT 推理session的option
	if err != nil {
		detInput.Destroy()
		for _, tensor := range detOutputs {
			tensor.Destroy()
		}
		return nil, fmt.Errorf("error creating ORT det session options: %w", err)
	}
	defer ortDetSO.Destroy()

	//to do add cuda or dml   err=options.AppendExecutionProvider .....
	// err = options.AppendExecutionProviderCoreML(0)
	// 	if err != nil {
	// 		inputTensor.Destroy()
	// 		outputTensor.Destroy()
	// 		return nil, fmt.Errorf("Error enabling CoreML: %w", err)
	// 	}

	// to do
	// get model input & output names & shapes

	detSession,err := ort.NewAdvancedSession(detModelPath,detInputName,detOutputNames,[]ort.Value{detInput},detOutputs, ortDetSO)
	if err != nil {
		detInput.Destroy()
		for _, tensor := range detOutputs {
			tensor.Destroy()
		}
		return nil, fmt.Errorf("error creating ORT session: %w", err)
	}
	
	arctinputShape := ort.NewShape(1, 3, face_align_image_size, face_align_image_size)
	arcInput,err := ort.NewEmptyTensor[float32](arctinputShape)
	if err != nil {
		return nil, fmt.Errorf("error creating recognition input tensor: %w", err)
	}
	arcOutputShape := ort.NewShape(1, 512)
	arcOutput, err := ort.NewEmptyTensor[float32](arcOutputShape)
	if err != nil {
		arcInput.Destroy()
		return nil, fmt.Errorf("error creating recognition output tensor: %w", err)
	}
	ortArcSO, err := ort.NewSessionOptions()     //ORT 推理session的option
	if err != nil {
		arcInput.Destroy()
		return nil, fmt.Errorf("error creating ORT arc session options: %w", err)
	}
	defer ortArcSO.Destroy()
	arcSession, err := ort.NewAdvancedSession(filepath.Join(onnxmodel_path, "w600k_r50.onnx"), arcInputName, arcOutputName, []ort.Value{arcInput}, []ort.Value{arcOutput}, ortArcSO)
	if err != nil {
		arcInput.Destroy()
		arcOutput.Destroy()
		return nil, fmt.Errorf("error creating Arcface ORT session: %w", err)
	}

	return &ModelSessions{
		detSession: detSession,
		arcSession: arcSession,
		detInput:     detInput,
		detOutputs:    detOutputs,
		arcInput:     arcInput,
		arcOutput:   arcOutput,
	}, nil
}



// Detect face in src image,
// return face boxes and landmarks, ordered by predict scores.
func FaceDetect(session *ModelSessions, src image.Image) ([][]float32, [][]float32, error) {
	Input:= session.detInput.GetData()
	det_scale,_ := preprocessImage(src, det_size, Input)
	// to deal err
	// face detect model inference
	err := session.detSession.Run()
	if err != nil {
		return nil, nil, err
	}
	// for _, output := range session.detOutputs {
	// 	output.(*ort.Tensor[float32]).GetData()
	// }
	dets, kpss := processResult(session.detOutputs, det_scale)

	return dets, kpss, nil
}

// Get face features by Arcface
// Parameter src is original image, lmk is face landmark detected by FaceDetect(),
// return features in a arrary, and norm_crop image
func FaceFeatures(session *ModelSessions,src image.Image, lmk []float32) ([]float32, image.Image, error) {
	// 裁剪对齐人脸
	aimg, err := norm_crop(src, lmk)
	if err!=nil {
		return nil, nil, err
	}
	Input:=session.arcInput.GetData()
	// save normalization crop face for test
	//_ = imaging.Save(aimg, "data/crop_norm.jpg")

	// prepare input data
	
	err = preprocessFace(aimg, face_align_image_size,Input)
	if err!=nil {
		return nil, nil, err
	}	

	// face features modle inference
	// res2, err := arcfaceModel.([]onnxruntime.TensorValue{
	// 	{
	// 		Value: input2,
	// 		Shape: shape2,
	// 	}
	// })
	// if err != nil {
	// 	return nil, nil, err
	// }
	err = session.arcSession.Run()
	if err != nil {
		return nil, nil, err
	}
	// if len(res2) == 0 {
	// 	return nil, nil, errors.New("Fail to get result")
	// }
	
	// return res2[0].Value.([]float32), aimg, nil
	return session.arcOutput.GetData(), aimg, nil
}