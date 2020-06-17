import datetime


class Timestamps:

    def iso_now_seconds() -> str:
        """Generate ISO format timestamp containing date and time up to the number of seconds.

        Drops milliscond information from timestamp.
        """
        return datetime.datetime.now().isoformat()[0:19]
