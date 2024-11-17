
class DateTimeUtils:

    @staticmethod
    def format_time(seconds):
        """
        Formata o tempo em segundos para o formato HH:MM:SS.mmm.

        :param seconds: Tempo em segundos a ser formatado.
        :return: String formatada representando o tempo.
        """
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"