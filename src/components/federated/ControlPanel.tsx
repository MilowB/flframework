import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Play, Square, RotateCcw, Zap, Hash } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ControlPanelProps {
  isRunning: boolean;
  currentRound: number;
  totalRounds: number;
  seed: number;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  onSeedChange: (seed: number) => void;
}

export const ControlPanel = ({
  isRunning,
  currentRound,
  totalRounds,
  seed,
  onStart,
  onStop,
  onReset,
  onSeedChange,
}: ControlPanelProps) => {
  const progress = totalRounds > 0 ? (currentRound / totalRounds) * 100 : 0;

  return (
    <div className="p-6 rounded-xl bg-gradient-card border border-border shadow-card">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
        {/* Left: Controls */}
        <div className="flex items-center gap-3">
          <Button
            onClick={isRunning ? onStop : onStart}
            className={cn(
              'gap-2 min-w-[140px] font-medium transition-all',
              isRunning 
                ? 'bg-destructive hover:bg-destructive/90' 
                : 'bg-gradient-primary hover:opacity-90'
            )}
          >
            {isRunning ? (
              <>
                <Square className="w-4 h-4" />
                Arrêter
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Démarrer
              </>
            )}
          </Button>
          
          <Button
            variant="outline"
            onClick={onReset}
            disabled={isRunning}
            className="gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            Réinitialiser
          </Button>
        </div>

        {/* Center: Progress */}
        <div className="flex-1 max-w-md">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-muted-foreground">Progression</span>
            <span className="font-mono text-sm">
              Round <span className="text-primary">{currentRound}</span> / {totalRounds}
            </span>
          </div>
          <div className="relative h-2 rounded-full bg-muted overflow-hidden">
            <div 
              className="absolute inset-y-0 left-0 bg-gradient-primary transition-all duration-300 rounded-full"
              style={{ width: `${progress}%` }}
            />
            {isRunning && (
              <div className="absolute inset-0 overflow-hidden">
                <div className="absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-[shimmer_2s_infinite] -translate-x-full" 
                  style={{ animation: 'shimmer 2s infinite' }}
                />
              </div>
            )}
          </div>
        </div>

        {/* Right: Seed */}
        <div className="flex items-center gap-4 min-w-[160px]">
          <Hash className="w-5 h-5 text-muted-foreground" />
          <div className="flex-1">
            <Label className="text-sm text-muted-foreground mb-1 block">Seed</Label>
            <Input
              type="number"
              min={0}
              step={1}
              value={seed}
              onChange={e => onSeedChange(Number(e.target.value))}
              disabled={isRunning}
              className="w-24 h-8"
            />
          </div>
        </div>

        {/* Status Indicator */}
        {isRunning && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 border border-primary/30">
            <Zap className="w-4 h-4 text-primary animate-pulse" />
            <span className="text-sm font-medium text-primary">En cours</span>
          </div>
        )}
      </div>
    </div>
  );
};