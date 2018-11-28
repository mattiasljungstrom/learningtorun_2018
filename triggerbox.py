class TriggerBox():
    def __init__(self, msg, texts, callbacks):
        def show():
            try:
                import pymsgbox
            except ImportError:
                return
            while True:
                selected = pymsgbox.confirm(text=msg, title='TriggerBox', buttons=texts)
                for i,t in enumerate(texts):
                    if t==selected:
                        callbacks[i]()
        import threading as th
        t = th.Thread(target=show)
        t.daemon = True
        t.start()
