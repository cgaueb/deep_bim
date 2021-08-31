#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <thread>
#include <atomic>

#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader/tiny_obj_loader.h>

struct Vox_Object {
    std::string name;
    std::vector<glm::vec3>  vertices;
    std::vector<glm::uvec3> faces;

    glm::mat4 transform = glm::mat4(1.0f);
    glm::vec3 bbox[2] = { {FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX} };
};

struct Vox_AABB {
    glm::vec3 lb;
    glm::vec3 ub;

    Vox_AABB() : lb(0.0), ub(0.0) {}
    Vox_AABB(glm::vec3 mi, glm::vec3 ma) : lb(mi), ub(ma) {}
    Vox_AABB(float* vals) : Vox_AABB(glm::vec3(vals[0], vals[1], vals[2]), glm::vec3(vals[3], vals[4], vals[5])) {}
    Vox_AABB(double* vals) : Vox_AABB(glm::vec3((float)vals[0], (float)vals[1], (float)vals[2]), glm::vec3((float)vals[3], (float)vals[4], (float)vals[5])) {}
    
    std::array<Vox_AABB, 8> split() const;
    glm::vec3 extents() const { return (ub - lb); }
    glm::vec3 center() const { return (ub + lb) / 2.0f; }
    float volume() const {
        glm::vec3 cell_extents = extents();
        return cell_extents.x * cell_extents.y * cell_extents.z;
    }
    float radius() const {
        glm::vec3 cell_extents = extents();
        float longest_value = glm::max(cell_extents.x, glm::max(cell_extents.y, cell_extents.z));
        return longest_value * 0.5f;
    }
    int longest() const {
        glm::vec3 cell_extents = extents();
        float longest_value = glm::max(cell_extents.x, glm::max(cell_extents.y, cell_extents.z));
        if (longest_value == cell_extents.x) return 0;
        if (longest_value == cell_extents.y) return 1;
        if (longest_value == cell_extents.z) return 2;
        return 0;
    }

    bool intersects(const glm::vec3& va, const glm::vec3& vb, const glm::vec3& vc) const
    {
        return intersects(Vox_AABB(glm::min(va, glm::min(vb, vc)), glm::max(va, glm::max(vb, vc))));
    }

    bool intersects(const Vox_AABB& B) const
    {
        //Check if this's max is greater than B's min and this's min is less than B's max
        return(
            ub.x > B.lb.x &&
            lb.x < B.ub.x&&
            ub.y > B.lb.y &&
            lb.y < B.ub.y&&
            ub.z > B.lb.z &&
            lb.z < B.ub.z
            );
    }
    bool contains(glm::vec3 point) const
    {
        return(
            ub.x >= point.x &&
            lb.x <= point.x &&
            ub.y >= point.y &&
            lb.y <= point.y &&
            ub.z >= point.z &&
            lb.z <= point.z
            );
    }
    bool singular(Vox_AABB B) const {
        return ((lb == B.lb) && (ub == B.ub));
    }
};

Vox_AABB intersection(const Vox_AABB& lhs, const Vox_AABB& rhs) {
    if (!lhs.intersects(rhs)) return {};
    return {
        { glm::max(lhs.lb.x, rhs.lb.x), glm::max(lhs.lb.y, rhs.lb.y), glm::max(lhs.lb.z, rhs.lb.z) },
        { glm::min(lhs.ub.x, rhs.ub.x), glm::min(lhs.ub.y, rhs.ub.y), glm::min(lhs.ub.z, rhs.ub.z) }
    };
}

Vox_AABB operator+ (const Vox_AABB& lhs, const glm::vec3& rhs) {
    return { lhs.lb + rhs, lhs.ub + rhs };
}

Vox_AABB operator- (const Vox_AABB& lhs, const glm::vec3& rhs) {
    return { lhs.lb - rhs, lhs.ub - rhs };
}

Vox_AABB operator/ (const Vox_AABB& lhs, const glm::uvec3& rhs) {
    return { lhs.lb / glm::vec3(rhs), lhs.ub / glm::vec3(rhs) };
}

std::array<Vox_AABB, 8> Vox_AABB::split() const {
    const auto cell = Vox_AABB{ lb, center() };
    const auto cell_extents = cell.extents();
    return {
        cell + glm::vec3(0,             0,             0),
        cell + glm::vec3(cell_extents.x,0,             0),
        cell + glm::vec3(0,             cell_extents.y,0),
        cell + glm::vec3(0,             0,             cell_extents.z),

        cell + glm::vec3(cell_extents.x,cell_extents.y,0),
        cell + glm::vec3(cell_extents.x,0             ,cell_extents.z),
        cell + glm::vec3(0,             cell_extents.y,cell_extents.z),
        cell + glm::vec3(cell_extents.x,cell_extents.y,cell_extents.z)
    };
}

/* Classify points whether they are inside or outside a mesh */
typedef enum {
    VOX_POINT_CLASS_INSIDE,
    VOX_POINT_CLASS_OUTSIDE
} Vox_PointClassification;

bool
Vox_ObjectLoad(Vox_Object* object, const char* filename);
bool
Vox_ObjectTransform(Vox_Object* object, const float* transformv);
bool
Vox_ObjectVoxelize(Vox_Object* object,
    unsigned int dim_x, unsigned int dim_y, unsigned int dim_z,
    unsigned int* binary_grid, float* voxel_grid,
    bool normAcrossLongest
);

bool
Vox_ObjectLoad(Vox_Object* object, const char* filename);
bool 
Vox_ObjectTransform(Vox_Object* object, const float* transformv);

void
Vox_VoxelizeFile(const std::string &, const std::string&, unsigned int, unsigned int, unsigned int, bool);
void
Vox_VoxelizeDirectory(const std::string &, const std::string&, unsigned int, unsigned int, unsigned int, bool);

int main(int argc, char* argv[])
{
    std::string op = "t";

    if (argc != 5) return 0;

    bool normAccrosLongest = false;

    if (!strcmp(argv[4], "1"))
    {
        normAccrosLongest = true;
    }
    else if (!strcmp(argv[4], "0"))
    {
        normAccrosLongest = false;
    }

    if (!strcmp(argv[1], "-d"))
    {
        Vox_VoxelizeDirectory(argv[2], argv[3], 32, 32, 32, normAccrosLongest);
    }
    else if (!strcmp(argv[1], "-f"))
    {
        Vox_VoxelizeFile(argv[2], argv[3], 32, 32, 32, normAccrosLongest);
    }

    return 0;
}


/* @see page 142 chapter 5 of
  * http://www.r-5.org/files/books/computers/algo-list/realtime-3d/Christer_Ericson-Real-Time_Collision_Detection-EN.pdf
  */
typedef glm::vec3 Point;
typedef glm::vec3 Vector;

Point ClosestPtPointTriangle(Point p, Point a, Point b, Point c) {
    // Check if P in vertex region outside A
    Vector ab = b - a;
    Vector ac = c - a;
    Vector ap = p - a;
    float d1 = glm::dot(ab, ap);
    float d2 = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a; // barycentric coordinates (1,0,0)

    // Check if P in vertex region outside B
    Vector bp = p - b;
    float d3 = glm::dot(ab, bp);
    float d4 = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b; // barycentric coordinates (0,1,0)

    // Check if P in edge region of AB, if so return projection of P onto AB
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + v * ab;// barycentric coordinates (1-v,v,0)
    }

    // Check if P in vertex region outside C
    Vector cp = p - c;
    float d5 = glm::dot(ab, cp);
    float d6 = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;// barycentric coordinates (0,0,1)

    // Check if P in edge region of AC, if so return projection of P onto AC
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + w * ac; // barycentric coordinates (1-w,0,w)
    }

    // Check if P in edge region of BC, if so return projection of P onto BC
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b); // barycentric coordinates (0,1-w,w)
    }

    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;

    return a + ab * v + ac * w; //=u*a+v*b+w*c,u=va*denom = 1.0f-v-w
}

float GetCLosestPointOnTriangle(const glm::vec3* vertices, unsigned int attr_count,
    const glm::uvec3* faces, unsigned int face_count, glm::vec3 query, float bias, glm::vec3* result, Vox_PointClassification* p_class) {

    auto min_class = VOX_POINT_CLASS_OUTSIDE;
    auto min_distance = 1000000.0f;
    auto min_face = faces[0];
    for (unsigned int face_index = 0; face_index < face_count; face_index++) {
        const auto& face = faces[face_index];
        const auto& va = vertices[face.x];
        const auto& vb = vertices[face.y];
        const auto& vc = vertices[face.z];

        const auto edge_a = vb - va;
        const auto edge_b = vc - va;
        const auto gnormal = glm::normalize(glm::cross(edge_a, edge_b));

        const auto closest_point = ClosestPtPointTriangle(query, va, vb, vc);
        const auto VtC = glm::normalize(query - va);
        const auto temp_class = (glm::dot(gnormal, VtC) >= 0.0) ? VOX_POINT_CLASS_OUTSIDE : VOX_POINT_CLASS_INSIDE;
        const auto dist = glm::distance(closest_point, query);

        /* This is an attempt to "favor" being on the outside
         * False positive voxels can be considered much more
         * problematic than false negatives
         */
        if (temp_class == VOX_POINT_CLASS_OUTSIDE && min_class == VOX_POINT_CLASS_INSIDE) {
            if (dist <= min_distance + bias) {
                min_distance = dist;
                min_face = face;
                min_class = temp_class;
                *result = closest_point;
            }
        }
        else {
            if (dist < min_distance) {
                min_distance = dist;
                min_face = face;
                min_class = temp_class;
                *result = closest_point;
            }
        }
    }

    const auto& face = min_face;
    const auto& va = vertices[face.x];
    const auto& vb = vertices[face.y];
    const auto& vc = vertices[face.z];

    const auto edge_a = vb - va;
    const auto edge_b = vc - va;
    const auto gnormal = glm::normalize(glm::cross(edge_a, edge_b));
    const auto VtC = glm::normalize(query - *result);

    *p_class = (glm::dot(gnormal, VtC) > 0.0) ? VOX_POINT_CLASS_OUTSIDE : VOX_POINT_CLASS_INSIDE;

    return min_distance;
}

bool
Vox_ObjectVoxelize(Vox_Object* object, 
    unsigned int dim_x, unsigned int dim_y, unsigned int dim_z,
    unsigned int* binary_grid, float* voxel_grid, bool normAcrossLongest
) {
    auto root = Vox_AABB(object->bbox[0], object->bbox[1]);
    auto aabb_size = root.extents();
    auto root_new = root;

    if (normAcrossLongest)
    {
        root_new = Vox_AABB(root.lb, root.lb + aabb_size[root.longest()]);
    }

    aabb_size = root_new.extents();
    auto cell_size = aabb_size / glm::vec3{ dim_x, dim_y, dim_z };

    auto origin = Vox_AABB(root.lb, root.lb + cell_size);

    if (normAcrossLongest)
    {
        origin = origin - (root_new.center() - root.center());
    }

    const auto  bias = cell_size[origin.longest()] / 100.0f;
    const auto& faces = object->faces;
    const auto& vertices = object->vertices;

    for (unsigned int i = 0; i < dim_x; i++) {
        for (unsigned int j = 0; j < dim_y; j++) {
            for (unsigned int k = 0; k < dim_z; k++) {
                unsigned int index = k * dim_y * dim_x + j * dim_x + i;
                auto class_cell = VOX_POINT_CLASS_OUTSIDE;

                const auto cell = origin + glm::vec3(i, j, k) * cell_size;
                const auto center = cell.center();

                glm::vec3 poi;
                float dist = GetCLosestPointOnTriangle(
                    vertices.data(), (unsigned int)vertices.size(),
                    faces.data(), (unsigned int)faces.size(), center, bias, &poi, &class_cell
                );

                binary_grid[index] = 0;
                if (cell.contains(poi) || class_cell == VOX_POINT_CLASS_INSIDE) {
                    binary_grid[index] = 1;
                    voxel_grid[3 * index + 0] = center.x;
                    voxel_grid[3 * index + 1] = center.y;
                    voxel_grid[3 * index + 2] = center.z;
                }
            }
        }
    }
    return true;
}

void
Vox_VoxelizeDirectory(const std::string & filepath, const std::string& outFolder, unsigned int dim_x, unsigned int dim_y, unsigned int dim_z, bool normAcrossLongest)
{
    HANDLE hFind = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATA ffd;
    char szDir[MAX_PATH];

    std::string dir_path = filepath;
    std::replace(dir_path.begin(), dir_path.end(), '\\', '/');

    strcpy(szDir, dir_path.c_str());
    strcat(szDir, "/*");

    hFind = FindFirstFile(szDir, &ffd);

    if (INVALID_HANDLE_VALUE == hFind)
        return;

    //CreateDirectoryA(outFolder.c_str(), NULL);

    std::vector<std::string> files;
    std::vector<std::thread> worker_pool;
    auto thread_count = std::thread::hardware_concurrency();
    std::atomic<int> next = 0;
    do
    {
        if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            std::string filename = dir_path + "/" + ffd.cFileName;
            if ((filename.find(".obj") == std::string::npos) && (filename.find(".OBJ") == std::string::npos))
                continue;
            files.push_back(filename);
        }
    } while (FindNextFile(hFind, &ffd) != 0);

    for (size_t worker_index = 0; worker_index < thread_count; worker_index++) {
        worker_pool.push_back(std::thread([&files, &outFolder, &next](unsigned int dim_x, unsigned int dim_y, unsigned int dim_z, bool normAcrossLongest) {
            auto index = next.fetch_add(1);
            while (index < files.size()) {
                Vox_VoxelizeFile(files[index].c_str(), outFolder, dim_x, dim_y, dim_z, normAcrossLongest);
                index = next.fetch_add(1);
            }
        }, dim_x, dim_y, dim_z, normAcrossLongest));
    }
    for (auto& worker : worker_pool) {
        worker.join();
    }
}

void
Vox_VoxelizeFile(const std::string & filepath, const std::string & outFolder, unsigned int dim_x, unsigned int dim_y, unsigned int dim_z, bool normAcrossLongest)
{
    Vox_Object A;

    const auto last_slash_idx = filepath.find_last_of("\\/");
    const auto last_dot_idx = filepath.find_last_of(".");
    if (std::string::npos == last_slash_idx || std::string::npos == last_dot_idx)
        return;

    auto filename_out = filepath.substr(last_slash_idx + 1, last_dot_idx - last_slash_idx - 1);
    filename_out += ".asc";

    if (!Vox_ObjectLoad(&A, filepath.c_str())) {
        return;
    }

    std::cout << "BC3D: Voxelizing " << filepath << " to [" << dim_x << ", " << dim_y << ", " << dim_z << "]"  << std::endl;

    std::vector<unsigned int> binary_grid;
    std::vector<glm::vec3>    voxel_grid;
    binary_grid.resize(dim_x * dim_y * dim_z);
    voxel_grid.resize(dim_x * dim_y * dim_z);

    Vox_ObjectVoxelize(&A, dim_x, dim_y, dim_z, binary_grid.data(), (float*)voxel_grid.data(), normAcrossLongest);

    FILE* f = fopen((outFolder + "/" + filename_out).c_str(), "w");
    for (unsigned int i = 0; i < dim_x; i++) {
        for (unsigned int j = 0; j < dim_y; j++) {
            for (unsigned int k = 0; k < dim_z; k++) {
                unsigned int index = k * dim_y * dim_x + j * dim_x + i;
                if (1 == binary_grid[index]) {
                    const auto& voxel = voxel_grid[index];
                    fprintf(f, "%d %d %d\n", i, j, k);
                }
            }
        }
    }

    fclose(f);
}

bool
Vox_ObjectLoad(Vox_Object* object, const char* filename) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename,
        "./", true);

    if (!ret) {
        return false;
    }

    unsigned int vertex_index = 0;
    unsigned int face_index = 0;
    for (const auto& shape : shapes) {
        unsigned int index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                std::cout << "Not a triangular mesh" << std::endl;
                continue;
            }

            object->faces.push_back({ vertex_index + 0, vertex_index + 1, vertex_index + 2 });
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                float vx = attrib.vertices[3 * idx.vertex_index + 0];
                float vy = attrib.vertices[3 * idx.vertex_index + 1];
                float vz = attrib.vertices[3 * idx.vertex_index + 2];
                object->vertices.push_back({ vx, vy, vz });
                vertex_index++;
            }
            index_offset += fv;
        }
    }

    object->bbox[0] = object->vertices[0];
    object->bbox[1] = object->vertices[0];

    for (const auto& vertex : object->vertices) {
        object->bbox[0] = glm::min(object->bbox[0], vertex);
        object->bbox[1] = glm::max(object->bbox[1], vertex);
    }

    object->transform = glm::mat4(1.0f);

    return true;
}

bool 
Vox_ObjectTransform(Vox_Object* object, const float* transformv) {
    glm::mat4 transform;
    memcpy(&transform[0][0], transformv, sizeof(transform));

    for (auto& vertex : object->vertices) {
        vertex = glm::vec3(transform * glm::vec4(vertex, 1.0f));
    }

    object->bbox[0] = glm::vec3(transform * glm::vec4(object->bbox[0], 1.0f));
    object->bbox[1] = glm::vec3(transform * glm::vec4(object->bbox[1], 1.0f));

    object->transform = transform;

    return true;
}
