#include <iostream>
#include <assert.h>
#include <setjmp.h>
#include <jpeglib.h>
#include <fstream>
#include <vector>

#define MAX(a,b) (a>=b?a:b);
#define MIN(a,b) (a<=b?a:b);

void write_csv(char *filename, std::vector<std::vector<int> > *ary){
    std::ofstream ofs(filename);
    for(int i=0; i<3; i++){
        for(int j=0; j<ary[i].size(); j++){
            for(int k=0; k<64; k++){
                if(k != 63){
                    ofs<<ary[i][j][k] << ",";
                }else{
                    ofs<< ary[i][j][k] << std::endl;
                }
            }
        }
    }
}

void dump_csv_coef(j_decompress_ptr cinfo, jvirt_barray_ptr *coeffs, int c, std::vector<std::vector<int> > *ary){
    jpeg_component_info *ci_ptr = &cinfo->comp_info[c];
    JBLOCKARRAY buf =
        (cinfo->mem->access_virt_barray)
        (
         (j_common_ptr)cinfo,
         coeffs[c],
         0,
         ci_ptr->v_samp_factor,
         FALSE
        );

    //one component block array.
    //it depends on width and heights
    std::vector<std::vector<int> > comps;

    // not jegzag but normal
    for(int sf=0; (JDIMENSION)sf < ci_ptr->height_in_blocks; ++sf){
        JDIMENSION b;
        for(b=0; b<ci_ptr->width_in_blocks; ++b){
            std::vector<int> one_line_ary; // one block
            one_line_ary.resize(64);
            for(int j=0; j<64; j++){
                int dc = buf[sf][b][j];
                dc = MAX(-128, dc);
                dc = MIN(128, dc);
                one_line_ary[j] = dc;
            }
            comps.push_back(one_line_ary);
        }
    }

    ary[c] = comps;
}

GLOBAL(int) read_JPEG_file(char *filename, std::vector<std::vector<int> > *ary){
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;

    if((infile = fopen(filename, "rb")) == NULL){
        fprintf(stderr, "can't open %s\n", filename);
        return 0;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    jpeg_read_header(&cinfo, TRUE);
    jvirt_barray_ptr *coeffs = jpeg_read_coefficients(&cinfo);

    for(int c=0; c<cinfo.num_components; c++){
        dump_csv_coef(&cinfo, coeffs, c, ary);
    }

    (void) jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 1;
}

int main(int argc, char **argv){
    int ret=0;
    if(argc != 3){
        fprintf(stderr, "usage: %s <jpg file> <output.csv>", argv[0]);
        return 1;
    }

    std::vector<std::vector<int> > ary[3];

    ret = read_JPEG_file(argv[1], ary);
    write_csv(argv[2], ary);

    return 0;
}
