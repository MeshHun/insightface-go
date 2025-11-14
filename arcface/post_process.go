package arcface

import (
	"fmt"
	"sort"

	ort "github.com/yalue/onnxruntime_go"
)

const (

)

var (
	nms_thresh = float32(0.4)
	det_thresh = float32(0.5)
	// len(outputs)==9
	_fmc = 3
	_feat_stride_fpn = []int{8, 16, 32}
	_num_anchors = 2
)

// process result after face-detect model inferenced
func processResult(net_outs []ort.Value, det_scale float32) ([][]float32, [][]float32) {
	//for i:=0;i<len(net_outs);i++ {
	//	log.Printf("Success do predict, shape : %+v, result : %+v\n", 
	//		net_outs[i].Shape, 
	//		net_outs[i].Value.([]float32)[:net_outs[i].Shape[1]], // only show one value
	//	)
	//}

	center_cache := make(map[string][][]float32)

	var scores_list []float32
	var bboxes_list [][]float32
	var kpss_list [][]float32

	for idx, stride := range _feat_stride_fpn {
		// stride := _feat_stride_fpn[idx]
		scores := net_outs[idx].(*ort.Tensor[float32]).GetData()
		bbox_preds := net_outs[idx+_fmc].(*ort.Tensor[float32]).GetData()
		// 边界框坐标缩放（x步长）
		for i := range bbox_preds { 					
			bbox_preds[i] = bbox_preds[i] * float32(stride)
		}

		// var kps_preds []float32 // landmark
		kps_preds := net_outs[idx+_fmc*2].(*ort.Tensor[float32]).GetData()
		for i := range kps_preds { 
			kps_preds[i] = kps_preds[i] * float32(stride)
		}
		// 计算特征图尺寸
		height := det_size / stride
		width := det_size / stride
		key := fmt.Sprintf("%d-%d-%d", height, width, stride)
		// 生成锚点中心（缓存复用）
		var anchor_centers [][]float32
		if val, ok := center_cache[key]; ok {
			anchor_centers = val
		} else {
			anchor_centers = make([][]float32, height*width*_num_anchors)
			for i:=0;i<height;i++ {
				for j:=0;j<width;j++ {
					for k:=0;k<_num_anchors;k++ {
						anchor_centers[i*width*_num_anchors+j*_num_anchors+k] = []float32{float32(j*stride), float32(i*stride)}
					}
				}
			}
			//log.Println(stride, len(anchor_centers), anchor_centers)

			if len(center_cache)<100 {
				center_cache[key] = anchor_centers
			}		
		}

		// filter by det_thresh == 0.5 过滤低置信度结果
		var pos_inds []int
		for i,score := range scores {
			if score > det_thresh {
				pos_inds = append(pos_inds, i)
			}
		}

		bboxes := distance2bbox(anchor_centers, bbox_preds)
		kpss := distance2kps(anchor_centers, kps_preds)
		//收集有效结果
		for _,i:=range pos_inds {
			scores_list = append(scores_list, scores[i])
			bboxes_list = append(bboxes_list, bboxes[i])
			kpss_list = append(kpss_list, kpss[i])
		}
	}


	// post process after get boxes and landmarks

	for i := range bboxes_list {
		for j:=0;j<4;j++ {
			bboxes_list[i][j] /= det_scale
		}
		bboxes_list[i] = append(bboxes_list[i], scores_list[i])

		for j:=0;j<10;j++ {
			kpss_list[i][j] /= det_scale
		}
		kpss_list[i] = append(kpss_list[i], scores_list[i])
	}

	sort.Slice(bboxes_list, func(i, j int) bool { return bboxes_list[i][4] > bboxes_list[j][4] })
	sort.Slice(kpss_list, func(i, j int) bool { return kpss_list[i][10] > kpss_list[j][10] })

	//非极大值抑制
	keep := nms(bboxes_list)

	//筛选最终结果
	det := make([][]float32, len(keep))
	kpss := make([][]float32, len(keep))
	for i, idx:= range keep {
		det[i] = bboxes_list[idx]
		kpss[i] = kpss_list[idx]
	}

	return det, kpss
}


func distance2bbox(points [][]float32, distance []float32) (ret [][]float32) {
	ret = make([][]float32, len(points))
	for i := range points {
		start := i * 4
		if start+3 >= len(distance) {
			fmt.Print("有越界风险")
			continue // 防止越界
		}
		ret[i] = []float32{
			points[i][0] - distance[i*4+0],
			points[i][1] - distance[i*4+1],
			points[i][0] + distance[i*4+2],
			points[i][1] + distance[i*4+3],
		}
	}
	return
}

func distance2kps(points [][]float32, distance []float32) (ret [][]float32) {
	ret = make([][]float32, len(points))
	for i := range points {
		ret[i] = make([]float32, 10)
		start := i * 10
		if start+9 >= len(distance) {
			fmt.Print("有越界风险")
			continue // 防止越界
		}
		for j:=0;j<10;j=j+2 {
			ret[i][j]   = points[i][j%2] + distance[i*10+j]
			ret[i][j+1] = points[i][j%2+1] + distance[i*10+j+1]
		} 
	}
	return
}


func max(a, b float32) float32 {
	if a>b { 
		return a
	} else {
		return b
	}
}

func min(a, b float32) float32 {
	if a<b { 
		return a
	} else {
		return b
	}
}

func nms(dets [][]float32) (ret []int) {
	if len(dets)==0 {
		return
	}

	var order []int
	areas := make([]float32, len(dets))
	for i, det := range dets {
		order = append(order, i)
		areas[i] = (det[2] - det[0] + 1) * (det[3] - det[1] + 1)
	}
	for len(order)>0 {
		i := order[0]
		ret = append(ret, i)

		var keep []int
		for _,j := range order[1:] {
			xx1 := max(dets[i][0], dets[j][0])
			yy1 := max(dets[i][1], dets[j][1])
			xx2 := min(dets[i][2], dets[j][2])
			yy2 := min(dets[i][3], dets[j][3])

			w := max(0.0, xx2 - xx1 + 1)
			h := max(0.0, yy2 - yy1 + 1)
			inter := w * h
			ovr := inter / (areas[i] + areas[j] - inter)

			if ovr <= nms_thresh {
				keep = append(keep, j)
			}
		}

		order = keep
	}
	return
}
