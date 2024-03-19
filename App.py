import cv2
import mediapipe as mp
import pygame

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

pygame.init()
screen = pygame.display.set_mode((800, 600))

color = (255, 0, 0)
distance_cm = float(input("Введите длину: ")) 

prev_x = None
prev_y = None

cap = cv2.VideoCapture(0)
running = True
while running:
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results.pose_landmarks is not None:
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

        middle_hip_x = (right_hip.x + left_hip.x) / 2
        middle_hip_y = (right_hip.y + left_hip.y) / 2

        image_height, image_width, _ = image.shape
        point_below_middlehip_y = middle_hip_y + (distance_cm / image_height)

        x = int(middle_hip_x * image_width)
        y = int(point_below_middlehip_y * image_height)

        if prev_x is not None and prev_y is not None:
            pygame.draw.line(screen, color, (prev_x, prev_y), (x, y), 5)

        prev_x = x
        prev_y = y

        cv2.circle(annotated_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    pygame.display.flip()
    cv2.imshow("камера", annotated_image)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    if cv2.waitKey(5) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
