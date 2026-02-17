#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>

#include <iostream>
#include <vector>
#include <cmath>

// ─────────────────────────────────────────────
//  Callbacks & globals
// ─────────────────────────────────────────────
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

const unsigned int SCR_WIDTH  = 1200;
const unsigned int SCR_HEIGHT = 800;

Camera camera(glm::vec3(0.0f, 0.0f, 8.0f));
float lastX      = SCR_WIDTH  / 2.0f;
float lastY      = SCR_HEIGHT / 2.0f;
bool  firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Animation state
bool  paused       = false;
float rotSpeed     = 0.4f;   // radians per second
bool  spacePressed = false;  // edge-detect for space

// ─────────────────────────────────────────────
//  Helix / tube parameters
// ─────────────────────────────────────────────
const float HELIX_RADIUS   = 1.4f;   // radius of the helix path
const float TUBE_RADIUS    = 0.09f;  // thickness of each strand
const float RUNG_RADIUS    = 0.055f; // thickness of connecting rungs
const int   HELIX_STEPS    = 240;    // points sampled along the helix
const int   TUBE_SEGS      = 14;     // circle subdivisions around the tube
const int   RUNG_SEGS      = 10;     // circle subdivisions around each rung
const float HELIX_HEIGHT   = 9.0f;   // total height
const float HELIX_TURNS    = 4.5f;   // number of full rotations
const int   RUNG_EVERY     = 16;     // place a rung every N helix steps

const float PI = 3.14159265358979f;

// ─────────────────────────────────────────────
//  Geometry helpers
// ─────────────────────────────────────────────

// Evaluate the helix position at parameter t ∈ [0, 1]
glm::vec3 helixPos(float t, float phaseOffset)
{
    float angle = t * HELIX_TURNS * 2.0f * PI + phaseOffset;
    float x = HELIX_RADIUS * cosf(angle);
    float y = (t - 0.5f) * HELIX_HEIGHT;   // centre at y=0
    float z = HELIX_RADIUS * sinf(angle);
    return glm::vec3(x, y, z);
}

// Build a tube mesh along the helix strand identified by phaseOffset.
// Outputs interleaved (position, normal) into `verts`, triangle indices into `inds`.
void buildStrand(float phaseOffset,
                 std::vector<float>&        verts,
                 std::vector<unsigned int>& inds)
{
    // We need one extra sample at each end for tangent estimation
    int N = HELIX_STEPS;

    // Pre-compute Frenet frames (position + normal + binormal)
    std::vector<glm::vec3> pos(N), tan(N), nor(N), bin(N);

    for (int i = 0; i < N; ++i)
    {
        float t = float(i) / float(N - 1);
        pos[i]  = helixPos(t, phaseOffset);
    }

    // Tangent via central differences
    for (int i = 0; i < N; ++i)
    {
        int prev = (i == 0)     ? 0     : i - 1;
        int next = (i == N - 1) ? N - 1 : i + 1;
        tan[i] = glm::normalize(pos[next] - pos[prev]);
    }

    // Seed the first normal: pick any vector not parallel to tan[0]
    glm::vec3 seed = (fabsf(tan[0].y) < 0.9f) ? glm::vec3(0, 1, 0)
                                                : glm::vec3(1, 0, 0);
    nor[0] = glm::normalize(glm::cross(glm::cross(tan[0], seed), tan[0]));
    bin[0] = glm::cross(tan[0], nor[0]);

    // Propagate frame (double-reflection / rotation-minimising)
    for (int i = 1; i < N; ++i)
    {
        glm::vec3 v1 = pos[i] - pos[i - 1];
        float c1 = glm::dot(v1, v1);
        if (c1 > 1e-10f)
        {
            glm::vec3 riL = nor[i-1] - (2.0f / c1) * glm::dot(v1, nor[i-1]) * v1;
            glm::vec3 tiL = tan[i-1] - (2.0f / c1) * glm::dot(v1, tan[i-1]) * v1;
            glm::vec3 v2  = tan[i] - tiL;
            float c2 = glm::dot(v2, v2);
            nor[i] = (c2 > 1e-10f) ? riL - (2.0f / c2) * glm::dot(v2, riL) * v2
                                    : riL;
            nor[i] = glm::normalize(nor[i]);
        }
        else
        {
            nor[i] = nor[i - 1];
        }
        bin[i] = glm::cross(tan[i], nor[i]);
    }

    int M = TUBE_SEGS;
    unsigned int baseIndex = (unsigned int)(verts.size() / 6);

    // Emit ring vertices at each helix step
    for (int i = 0; i < N; ++i)
    {
        for (int s = 0; s <= M; ++s)
        {
            float a  = float(s) / float(M) * 2.0f * PI;
            float ca = cosf(a), sa = sinf(a);

            glm::vec3 normal = glm::normalize(ca * nor[i] + sa * bin[i]);
            glm::vec3 p      = pos[i] + TUBE_RADIUS * normal;

            verts.push_back(p.x);      verts.push_back(p.y);      verts.push_back(p.z);
            verts.push_back(normal.x); verts.push_back(normal.y); verts.push_back(normal.z);
        }
    }

    // Stitch rings into triangles
    for (int i = 0; i < N - 1; ++i)
    {
        for (int s = 0; s <= M; ++s)
        {
            unsigned int cur  = baseIndex + i       * (M + 1) + s;
            unsigned int next = baseIndex + (i + 1) * (M + 1) + s;

            inds.push_back(cur);
            inds.push_back(next);
            inds.push_back(cur + 1 < baseIndex + (i + 1) * (M + 1)
                               ? cur + 1 : baseIndex + i * (M + 1));

            inds.push_back(next);
            inds.push_back(next + 1 < baseIndex + (i + 2) * (M + 1)
                               ? next + 1 : baseIndex + (i + 1) * (M + 1));
            inds.push_back(cur + 1 < baseIndex + (i + 1) * (M + 1)
                               ? cur + 1 : baseIndex + i * (M + 1));
        }
    }
}

// Build cylinder mesh between two world-space points A and B.
void buildCylinder(const glm::vec3& A, const glm::vec3& B,
                   std::vector<float>&        verts,
                   std::vector<unsigned int>& inds)
{
    glm::vec3 axis = B - A;
    float len = glm::length(axis);
    if (len < 1e-5f) return;

    glm::vec3 axisN = axis / len;

    // Build perpendicular frame
    glm::vec3 seed  = (fabsf(axisN.y) < 0.9f) ? glm::vec3(0, 1, 0)
                                                : glm::vec3(1, 0, 0);
    glm::vec3 right = glm::normalize(glm::cross(axisN, seed));
    glm::vec3 up    = glm::cross(axisN, right);

    int M = RUNG_SEGS;
    unsigned int base = (unsigned int)(verts.size() / 6);

    // Two rings (at A and B)
    for (int ring = 0; ring < 2; ++ring)
    {
        glm::vec3 centre = (ring == 0) ? A : B;
        for (int s = 0; s <= M; ++s)
        {
            float a  = float(s) / float(M) * 2.0f * PI;
            glm::vec3 normal = cosf(a) * right + sinf(a) * up;
            glm::vec3 p      = centre + RUNG_RADIUS * normal;

            verts.push_back(p.x);      verts.push_back(p.y);      verts.push_back(p.z);
            verts.push_back(normal.x); verts.push_back(normal.y); verts.push_back(normal.z);
        }
    }

    // Side triangles
    for (int s = 0; s < M; ++s)
    {
        unsigned int a0 = base + s;
        unsigned int a1 = base + s + 1;
        unsigned int b0 = base + (M + 1) + s;
        unsigned int b1 = base + (M + 1) + s + 1;

        inds.push_back(a0); inds.push_back(b0); inds.push_back(a1);
        inds.push_back(a1); inds.push_back(b0); inds.push_back(b1);
    }

    // End caps (simple fan)
    // Cap A
    unsigned int centreA = (unsigned int)(verts.size() / 6);
    verts.push_back(A.x); verts.push_back(A.y); verts.push_back(A.z);
    verts.push_back(-axisN.x); verts.push_back(-axisN.y); verts.push_back(-axisN.z);
    for (int s = 0; s < M; ++s)
    {
        inds.push_back(centreA);
        inds.push_back(base + s);
        inds.push_back(base + s + 1);
    }

    // Cap B
    unsigned int centreB = (unsigned int)(verts.size() / 6);
    verts.push_back(B.x); verts.push_back(B.y); verts.push_back(B.z);
    verts.push_back(axisN.x); verts.push_back(axisN.y); verts.push_back(axisN.z);
    unsigned int ringB = base + (M + 1);
    for (int s = 0; s < M; ++s)
    {
        inds.push_back(centreB);
        inds.push_back(ringB + s + 1);
        inds.push_back(ringB + s);
    }
}

// Build ALL rungs into one merged mesh
void buildRungs(std::vector<float>&        verts,
                std::vector<unsigned int>& inds)
{
    int N = HELIX_STEPS;
    for (int i = 0; i < N; i += RUNG_EVERY)
    {
        float t = float(i) / float(N - 1);
        glm::vec3 A = helixPos(t, 0.0f);
        glm::vec3 B = helixPos(t, PI);
        buildCylinder(A, B, verts, inds);
    }
}

// Upload geometry to GPU, return VAO id
unsigned int uploadMesh(const std::vector<float>&        verts,
                        const std::vector<unsigned int>& inds,
                        unsigned int& vbo, unsigned int& ebo)
{
    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 verts.size() * sizeof(float),
                 verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 inds.size() * sizeof(unsigned int),
                 inds.data(), GL_STATIC_DRAW);

    // position (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // normal (location 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    return vao;
}

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main()
{
    // GLFW init
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);   // MSAA

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT,
                                          "DNA Double Helix", NULL, NULL);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    // ── Shader ──────────────────────────────
    Shader shader("dna_helix.vs", "dna_helix.fs");

    // ── Build geometry ───────────────────────
    std::vector<float>        vertsA, vertsB, vertsR;
    std::vector<unsigned int> indsA,  indsB,  indsR;

    buildStrand(0.0f, vertsA, indsA);      // Strand A  (cyan)
    buildStrand(PI,   vertsB, indsB);      // Strand B  (orange)
    buildRungs(vertsR, indsR);             // Rungs      (white)

    unsigned int vboA, eboA, vaoA = uploadMesh(vertsA, indsA, vboA, eboA);
    unsigned int vboB, eboB, vaoB = uploadMesh(vertsB, indsB, vboB, eboB);
    unsigned int vboR, eboR, vaoR = uploadMesh(vertsR, indsR, vboR, eboR);

    // ── Light setup ──────────────────────────
    glm::vec3 lightPos(5.0f, 8.0f, 5.0f);
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);

    // ── Colors ───────────────────────────────
    glm::vec3 colorA(0.18f, 0.80f, 0.90f);   // cyan
    glm::vec3 colorB(1.00f, 0.45f, 0.15f);   // orange
    glm::vec3 colorR(0.90f, 0.90f, 0.90f);   // near-white

    // Accumulated rotation angle (persists across frames)
    float rotAngle = 0.0f;

    // ── Render loop ──────────────────────────
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        if (!paused) rotAngle += rotSpeed * deltaTime;

        glClearColor(0.04f, 0.04f, 0.08f, 1.0f);   // near-black background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();

        // Matrices
        glm::mat4 projection = glm::perspective(
            glm::radians(camera.Zoom),
            float(SCR_WIDTH) / float(SCR_HEIGHT),
            0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();

        shader.setMat4("projection", projection);
        shader.setMat4("view",       view);
        shader.setVec3("lightPos",   lightPos);
        shader.setVec3("lightColor", lightColor);
        shader.setVec3("viewPos",    camera.Position);

        // Model: rotate the whole helix around Y
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, rotAngle, glm::vec3(0.0f, 1.0f, 0.0f));

        shader.setMat4("model", model);

        // Draw strand A
        shader.setVec3("objectColor", colorA);
        glBindVertexArray(vaoA);
        glDrawElements(GL_TRIANGLES, (GLsizei)indsA.size(), GL_UNSIGNED_INT, 0);

        // Draw strand B
        shader.setVec3("objectColor", colorB);
        glBindVertexArray(vaoB);
        glDrawElements(GL_TRIANGLES, (GLsizei)indsB.size(), GL_UNSIGNED_INT, 0);

        // Draw rungs
        shader.setVec3("objectColor", colorR);
        glBindVertexArray(vaoR);
        glDrawElements(GL_TRIANGLES, (GLsizei)indsR.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &vaoA); glDeleteBuffers(1, &vboA); glDeleteBuffers(1, &eboA);
    glDeleteVertexArrays(1, &vaoB); glDeleteBuffers(1, &vboB); glDeleteBuffers(1, &eboB);
    glDeleteVertexArrays(1, &vaoR); glDeleteBuffers(1, &vboR); glDeleteBuffers(1, &eboR);

    glfwTerminate();
    return 0;
}

// ─────────────────────────────────────────────
//  Callbacks
// ─────────────────────────────────────────────
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Camera movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD,  deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT,     deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT,    deltaTime);

    // Pause / resume — edge detect so one press = one toggle
    bool spaceDown = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
    if (spaceDown && !spacePressed) paused = !paused;
    spacePressed = spaceDown;

    // Speed control
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        rotSpeed = std::min(rotSpeed + 0.01f, 3.0f);
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        rotSpeed = std::max(rotSpeed - 0.01f, 0.0f);
}

void framebuffer_size_callback(GLFWwindow*, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow*, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }

    float xoffset =  xpos - lastX;
    float yoffset =  lastY - ypos;
    lastX = xpos; lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow*, double, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
