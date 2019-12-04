import sc2


class SimpleSC2BotAI(sc2.BotAI):
    """
        an dumb agent that do nothing
    """

    def __init__(self):
        super(SimpleSC2BotAI, self).__init__()

    async def on_step(self, action_list=None):
        await self.do_actions(action_list)
