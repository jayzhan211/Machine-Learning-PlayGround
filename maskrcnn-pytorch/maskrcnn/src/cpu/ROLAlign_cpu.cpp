#include "cpu/vision.h"

template<typename T>
struct PreCalc{
    int pos1;
    int pos2;
    int pos3;
    int pos4;
    T w1;
    T w2;
    T w3;
    T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc){
        int pre_calc_index = 0;
        for(int ph = 0; ph < pooled_height ; ph++){
            for (int pw = 0; pw < pooled_width; pw++) {
                 for (int iy = 0; iy < iy_upper; iy++) {
                    const T yy = roi_start_h + ph * bin_size_h +
                        static_cast<T>(iy + .5f) * bin_size_h  /
                            static_cast<T>(roi_bin_grid_h);
                    for (int ix = 0; ix < ix_upper; ix++) {
                        const T xx = roi_start_w + pw * bin_size_w +
                            static_cast<T>(ix + .5f) * bin_size_w /
                                static_cast<T>(roi_bin_grid_w);
                        T x = xx;
                        T y = yy;
                         // deal with elements out of featrue map
                        if (y < -1.0 || y > height || x < -1.0 || x > width) {
                            PreCalc<T> pc;
                            pc.pos1 = 0;
                            pc.pos2 = 0;
                            pc.pos3 = 0;
                            pc.pos4 = 0;
                            pc.w1 = 0;
                            pc.w2 = 0;
                            pc.w3 = 0;
                            pc.w4 = 0;
                            pre_calc[pre_calc_index] = pc;
                            pre_calc_index += 1;
                            continue;
                         }

                        if(y <= 0) y = 0;
                        if(x <= 0) x = 0;

                        int y_low = (int)y;
                        int x_low = (int)x;
                        int y_high;
                        int x_high;

                        if (y_low >= height - 1) {
                            y_high = y_low = height - 1;
                            y = (T)y_low;
                        }
                        else {
                            y_high = y_low + 1;
                        }

                        if (x_low >= width - 1) {
                            x_high = x_low = width - 1;
                            x = (T)x_low;
                        }
                        else {
                            x_high = x_low + 1;
                        }
                        T ly = y - y_low;
                        T lx = x - x_low;
                        T hy = 1. - ly, hx = 1. - lx;
                        T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

                        PreCalc<T> pc;
                        pc.pos1 = y_low * width + x_low;
                        pc.pos2 = y_low * width + x_high;
                        pc.pos3 = y_high * width + x_low;
                        pc.pos4 = y_high * width + x_high;
                        pc.w1 = w1;
                        pc.w2 = w2;
                        pc.w3 = w3;
                        pc.w4 = w4;
                        pre_calc[pre_calc_index] = pc;

                        pre_calc_index += 1;


                    }
                 }
            }
        }
    }

template <typename T>
void ROIAlignForward_cpu_kernel

