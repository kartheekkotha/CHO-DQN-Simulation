from utils.misc import gigahertz_to_megahertz, kmph_to_mpms

class ParamConfig:
    def __init__(self, config):
        """Initialize the parameter configuration with given dictionary values."""
        self.TRAIN = config.get("TRAIN", False)
        self.TOTAL_DISTANCE = config.get("TOTAL_DISTANCE", 3000)  # in meters
        self.TICKER_INTERVAL = config.get("TICKER_INTERVAL", 20)  # in ms
        
        # File Save Path
        self.SAVE_PATH = config.get("SAVE_PATH", 'results_temp')
        self.TEST_OUTPUT_FOLDER = config.get("TEST_OUTPUT_FOLDER", 'results_simulation')

        # Base Station - Shadowing Scenario
        self.FADING = config.get("FADING", True)
        self.POW_AVG = config.get("POW_AVG", True)
        self.AVG_COUNT = config.get("AVG_COUNT", 5)
        # Base Station - Dynamic Obstacles 
        self.OBSTACLES = config.get("OBSTACLES", True)
        self.EPSILON = config.get("EPSILON", 0.8)  

        # Other UE Effects
        self.CHECK_EFFECT = config.get("CHECK_EFFECT", False)
        self.LOAD_FACTOR = config.get("LOAD_FACTOR", 1)
        self.NOISE = config.get("NOISE", 5)  # dBm
        self.DYNAMIC = config.get("DYNAMIC", False)

        # Path Loss Model Parameters
        self.PATH_LOSS_EXPONENT = config.get("PATH_LOSS_EXPONENT", 2.8)
        self.REFERENCE_DISTANCE = config.get("REFERENCE_DISTANCE", 1)  # meters
        self.FREQ_NR = config.get("FREQ_NR", gigahertz_to_megahertz(25))  # MHz
        self.PTX = config.get("PTX", 2000)  # mW
        self.MAX_PTX = config.get("MAX_PTX", 7000)  # mW

        # User Equipment Parameters
        self.MIN_SPEED = config.get("MIN_SPEED", kmph_to_mpms(40))
        self.MAX_SPEED = config.get("MAX_SPEED", kmph_to_mpms(40))

        # Handover / Radio Link Failure (RLF) Parameters
        self.RLF_THRESHOLD = config.get("RLF_THRESHOLD", -67.5)  # dBm
        # for without fading [-33,-41] - optimal : -34 ~ -35
        # for fading [-73,-82] - optimal : -78 ~ -80
        self.N310_THRESHOLD = config.get("N310_THRESHOLD", 6)
        self.T310 = config.get("T310", 1000) #100~150
        self.HANDOVER_FAILURE_RATE = config.get("HANDOVER_FAILURE_RATE", 0.10)
        self.Tprep = config.get("Tprep", 100)  # Handover preparation time in ms - 100
        self.Texec = config.get("Texec", 80)  # Handover execution time in ms - 80
        self.Oprep = config.get("Oprep", 1)  # dB offset for handover preparation - 6
        self.Oexec = config.get("Oexec", 6)  # dB offset for handover execution - 1

        # SINR Calculation
        self.CALC_SINR = config.get("CALC_SINR", True)
        self.SINR_COUNT = config.get("SINR_COUNT", 10)

        self.INCREASE_OHO = config.get("INCREASE_OHO", True)
        # Agent Parameters
        self.USE_AGENT = config.get("USE_AGENT", False)
        self.AGENT = config.get("AGENT", 4)
        #4 for the main 10 parameter agent
        self.AGENT_PATH = config.get("AGENT_PATH", 'model_weights/dqn_agent_main.pth')
        self.POWER_INCREASE_QUANTITY = config.get("POWER_INCREASE_QUANTITY", 500)
        self.POWER_INCREASE_COUNT_THRESHOLD = config.get("POWER_INCREASE_COUNT_THRESHOLD", 3)
        self.POWER_COUNT_RESET_TIMER = config.get("POWER_COUNT_RESET_TIMER", 500)  # in ms
        self.COOL_DOWN = config.get("COOL_DOWN", True)
        self.COOL_DOWN_PERIOD = config.get("COOL_DOWN_PERIOD", 3000)

        # reward mechanism
        self.INCLUDE_SINR = config.get("INCLUDE_SINR", True)


