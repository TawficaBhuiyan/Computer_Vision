def __init__(self, cfg):
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.person_smoother = EMASmoother(cfg.smoothing.ema_alpha)
        self.ball = BallStabilizer(max_lost=cfg.smoothing.ball_max_lost)
        self.id_fix = IDStabilizer(max_lost=150)   # <-- add this
        self.viz = Visualizer(cfg)