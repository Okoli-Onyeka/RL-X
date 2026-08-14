{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# rate limiter\n",
    "\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "class CommandSmoother:\n",
    "    def __init__(\n",
    "        self,\n",
    "        dt,\n",
    "        max_rate=np.array([1.0, 1.0, 1.5]),\n",
    "    ):\n",
    "        self.dt = dt\n",
    "        self.max_rate = np.asarray(max_rate, dtype=np.float32)\n",
    "        self.current = np.zeros(3, dtype=np.float32)\n",
    "\n",
    "    def reset(self):\n",
    "        self.current[:] = 0.0\n",
    "\n",
    "    def update(self, target):\n",
    "        target = np.asarray(target, dtype=np.float32)\n",
    "\n",
    "        error = target - self.current\n",
    "\n",
    "        max_change = self.max_rate * self.dt\n",
    "\n",
    "        change = np.clip(\n",
    "            error,\n",
    "            -max_change,\n",
    "            max_change,\n",
    "        )\n",
    "\n",
    "        self.current += change\n",
    "\n",
    "        return self.current.copy()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
