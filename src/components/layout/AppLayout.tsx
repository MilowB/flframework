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
        {children}
      </SidebarInset>
    </SidebarProvider>
  );
}