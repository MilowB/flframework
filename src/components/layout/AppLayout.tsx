import { useState } from 'react';
import { SidebarProvider, SidebarInset, SidebarTrigger } from '@/components/ui/sidebar';
import { AppSidebar } from './AppSidebar';

interface AppLayoutProps {
  children: React.ReactNode;
}

export function AppLayout({ children }: AppLayoutProps) {
  const [open, setOpen] = useState(false);

  const handleNavigation = () => {
    setOpen(false);
  };

  return (
    <SidebarProvider open={open} onOpenChange={setOpen}>
      <AppSidebar onNavigate={handleNavigation} />
      <SidebarInset>
        <header className="sticky top-0 z-40 flex h-12 items-center gap-4 border-b border-border bg-background/80 backdrop-blur-lg px-4">
          <SidebarTrigger />
          <span className="text-sm font-medium text-muted-foreground">FL Framework</span>
        </header>
        {children}
      </SidebarInset>
    </SidebarProvider>
  );
}