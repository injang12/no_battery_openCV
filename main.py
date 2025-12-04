from app.ui.main_ui import CircleDetectorApp

def main():
    app = CircleDetectorApp()

    def on_resize(event):             # 창 크기 변경 이벤트에 따라 두 이미지 디스플레이를 모두 업데이트하도록 바인딩
        app.display_original_image()
        app.display_processed_image()

    app.bind('<Configure>', on_resize)
    app.mainloop()

if __name__ == "__main__":
    main()