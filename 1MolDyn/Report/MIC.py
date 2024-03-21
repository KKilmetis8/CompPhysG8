diff = self.pos[j] - particle.pos[j]
diff -= boxL * round(diff * c.inv_boxL,0) # 0 in, 1 out